"""
BLENDER LIVE  -  photoreal view that follows the running MuJoCo sim in real time (EEVEE).

MuJoCo still does all the physics. The sim (press B in its window) writes the scene (terrain, boulders, sun,
cube, sensor rig) to recordings/live_scene.npz, starts Blender with this script, and streams the cube pose,
wheel angles and sensor joints over UDP every frame. This script builds the same scene with render_blender.py's
builders (regolith shader, pebbles, faceted boulders, 162 W/m^2 sun, AgX exposure) and moves the objects.

    Blender --python blender_live.py -- --scene recordings/live_scene.npz --port 50777        (started by the sim)
    Blender -b --python blender_live.py -- --scene ... --port ... --test out.png              (headless check)

For the full path-traced film use render_blender.py on a recorded take instead.
"""
import argparse
import math
import os
import socket
import struct
import sys
import time

import bpy  # noqa: I001
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_blender as RB  # noqa: E402

CHASE = dict(dist=0.85, elev_deg=12.0, az_deg=35.0, lens=35.0)


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--test", default=None, help="headless: wait for poses, render one EEVEE frame here, quit")
    ap.add_argument("--pebbles", type=int, default=6000)
    return ap.parse_args(argv)


class Live:
    def __init__(self, a):
        R = dict(np.load(a.scene, allow_pickle=False))
        self.R = R
        q = dict(RB.QUALITY["preview"], pebbles=a.pebbles, scale=100)
        RB.clear()
        args = argparse.Namespace(res="1600x900", fps=30, engine="eevee", samples=None)
        RB.setup_render(args, q)
        sc = bpy.context.scene
        sc.render.use_motion_blur = False
        for attr, val in (("use_gtao", True), ("use_shadows", True), ("shadow_ray_count", 2),
                          ("use_raytracing", True)):                  # EEVEE quality knobs (if this build has them)
            if hasattr(sc.eevee, attr):
                try:
                    setattr(sc.eevee, attr, val)
                except (TypeError, AttributeError):
                    pass
        RB.make_world("faint")
        RB.make_sun(R["sun"])
        RB.build_surface(R, q)
        # The film shader's 0.5 mm bump relief turns most normals away from the low sun at EEVEE's few samples
        # (measured: mean frame brightness 6/255 with it, 41/255 without). Cycles averages it out; the live
        # view turns the bump off. The geometry (displacement, pebbles) and the film itself are unchanged.
        for mat in bpy.data.materials:
            if mat.node_tree is None:
                continue
            for n in mat.node_tree.nodes:
                if n.bl_idname == "ShaderNodeBump":
                    n.inputs["Strength"].default_value = 0.0
        self.cube, self.wheels = RB.build_cube(R)
        self.joints = RB.build_sensors(R, self.cube)
        self.names = [str(n) for n in R.get("sensor_joints", [])]
        nsens = int(np.asarray(R["sensor_q"]).shape[-1]) if "sensor_q" in R and np.size(R["sensor_q"]) else 0
        self.packet = f"<d3d4d3d{nsens}d"               # sim time, pos, quat, wheel angles, flat sensor joints
        self.packet_size = struct.calcsize(self.packet)
        cd = bpy.data.cameras.new("LiveCam")
        cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = CHASE["lens"], 36.0, 0.005, 3000.0
        self.cam = RB.link(bpy.data.objects.new("LiveCam", cd))
        self.cam.rotation_mode = "QUATERNION"
        sc.camera = self.cam
        self.target = Vector(R["pos"][-1]) if len(R["pos"]) else Vector((0, 0, 0))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", a.port))
        self.sock.setblocking(False)
        self.packets = 0
        self.apply(None)

    def poll(self):
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            if len(data) == self.packet_size:
                latest = struct.unpack(self.packet, data)
        if latest is not None:
            self.packets += 1
            self.apply(latest)
        return latest is not None

    def apply(self, p):
        if p is not None:
            pos, quat, wheel, sens = p[1:4], p[4:8], p[8:11], p[11:]
            self.cube.location = Vector(pos)
            self.cube.rotation_quaternion = Quaternion(quat)
            for k, wh in enumerate(self.wheels):
                wh.rotation_euler = (0, 0, float(wheel[k]))
            RB.pose_sensors(self.joints, self.names, sens)
            self.target = self.target.lerp(Vector(pos), 0.15)        # smooth chase, like the MuJoCo window
        tgt = self.target + Vector((0, 0, 0.035))
        az, el = math.radians(CHASE["az_deg"]), math.radians(CHASE["elev_deg"])
        eye = tgt + CHASE["dist"] * Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        eye.z = max(eye.z, RB.ground_h(self.R, eye.x, eye.y) + 0.025)
        self.cam.location = eye
        self.cam.rotation_quaternion = (tgt - eye).to_track_quat("-Z", "Y")


def show_camera_view():
    """Make every 3D viewport look through the chase camera with EEVEE 'Rendered' shading."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        space.region_3d.view_perspective = "CAMERA"
                        space.overlay.show_overlays = False
                        space.show_gizmo = False


def main():
    a = parse()
    live = Live(a)
    if a.test:                                             # headless check: real streamed poses -> one frame
        t0 = time.time()
        while live.packets < 30 and time.time() - t0 < 60:
            live.poll()
            time.sleep(0.01)
        bpy.context.scene.render.filepath = a.test
        bpy.ops.render.render(write_still=True)
        print(f"blender_live test: {live.packets} poses received, wrote {a.test}", flush=True)
        return

    def tick():
        live.poll()
        return 1 / 30
    bpy.app.timers.register(tick, persistent=True)
    bpy.app.timers.register(lambda: (show_camera_view(), None)[1], first_interval=0.5)
    print("blender_live: following the sim (close this window any time; the sim keeps running)", flush=True)


main()
