"""
RENDER_BLENDER  -  photoreal film of a recorded Psyche Cubli take (Blender 4.2+, Cycles path tracing).

The physics comes from MuJoCo (psyche_cubli_v2.py record, or press V in the 3D window).
This script only re-renders that exact motion photorealistically; it never changes it.

    BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
    $BLENDER -b -P render_blender.py -- --rec recordings/demo.npz --quality preview     # fast check
    $BLENDER -b -P render_blender.py -- --rec recordings/demo.npz --quality final --intro
    $BLENDER -b -P render_blender.py -- --rec recordings/demo.npz --still 20            # one frame at sim t=20 s
    python psyche_cubli_v2.py video --frames renders/demo/frames --out renders/demo.mp4

Options: --speed (sim seconds per video second, 1 = true real time), --start/--end (sim seconds),
--cam auto|cine|low|wide, --stars faint|none|cinematic, --res 1920x1080, --save-blend, --engine cycles|eevee

What is physical: sunlight is 162 W/m^2 (Psyche at ~2.9 AU) from a 0.18 deg sun disc (razor shadows),
no atmosphere (no sky light, black shadows lit only by bounce light from the ground), dust flies in clean
ballistic arcs under Psyche's gravity, wheel motion blur comes from the real wheel speed.
What is artistic: regolith/boulder micro-detail is procedural (no lander has imaged Psyche's surface yet),
camera moves, and stars ("faint" by default; a real camera exposed for sunlit ground shows none: --stars none).
"""
import argparse
import json
import math
import os
import sys

import bpy  # noqa: I001  (bpy first: bmesh/mathutils load with it)
import bmesh
import numpy as np
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

QUALITY = {
    "preview": dict(samples=24, scale=50, pebbles=8000, bounces=3),
    "final": dict(samples=160, scale=100, pebbles=60000, bounces=6),
    "film": dict(samples=600, scale=100, pebbles=120000, bounces=8),
}
SUN_IRRADIANCE = 162.0     # W/m^2  solar constant 1361 / 2.9^2 (Psyche's mean distance ~2.9 AU)
SUN_ANGLE_DEG = 0.18       # deg    the Sun's apparent diameter from 2.9 AU
EXPOSURE = -5.1            #        camera exposure (stops) for a 162 W/m^2 sun under AgX


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=os.path.join(HERE, "recordings", "demo.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quality", default="final", choices=list(QUALITY))
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--still", type=float, default=None, help="render one frame at this sim time")
    ap.add_argument("--cam", default="auto", choices=["auto", "cine", "low", "wide"])
    ap.add_argument("--stars", default="faint", choices=["none", "faint", "cinematic"])
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--intro", action="store_true", help="add an orbit-to-surface shot of Psyche first")
    ap.add_argument("--intro-seconds", type=float, default=6.0)
    ap.add_argument("--engine", default="cycles", choices=["cycles", "eevee"])
    ap.add_argument("--save-blend", action="store_true")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--only-frame", type=int, default=None, help="render just this frame of the clip (testing)")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(HERE, "renders", os.path.splitext(os.path.basename(a.rec))[0])
    return a


# ---------------------------------------------------------------------
# scene + geometry helpers
# ---------------------------------------------------------------------
def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_obj(name, V, F, smooth=False, sharp_deg=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(V, float).tolist(), [], np.asarray(F, int).tolist())
    me.validate()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
        if sharp_deg is not None and hasattr(me, "set_sharp_from_angle"):
            me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    me.update()
    return link(bpy.data.objects.new(name, me))


def read_stl(path):
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        rec = np.frombuffer(f.read(n * 50), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    V, inv = np.unique(np.round(tri, 7), axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)


def icosphere(name, subdiv=4, radius=1.0):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    return link(bpy.data.objects.new(name, me))


def tex(name, kind, **kw):
    t = bpy.data.textures.new(name, kind)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def displace(obj, texture, strength, mid=0.5, coords="LOCAL"):
    m = obj.modifiers.new("disp", "DISPLACE")
    m.texture, m.strength, m.mid_level, m.texture_coords = texture, strength, mid, coords
    return m


def subsurf(obj, render=2, view=0, simple=True):
    m = obj.modifiers.new("sub", "SUBSURF")
    m.levels, m.render_levels = view, render
    if simple:
        m.subdivision_type = "SIMPLE"
    return m


# ---------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------
def principled(mat):
    mat.use_nodes = True
    return mat.node_tree.nodes.get("Principled BSDF"), mat.node_tree


def inp(node, *names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    raise KeyError(names)


def mat_regolith(name="regolith", albedo=(0.115, 0.108, 0.100), lighter=(0.19, 0.18, 0.168)):
    """Psyche: optically dark-grey, radar-bright, metal-rich (FeNi grains glint in the sun)."""
    mat = bpy.data.materials.new(name)
    bsdf, nt = principled(mat)
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    big = N.new("ShaderNodeTexNoise")
    big.inputs["Scale"].default_value = 0.6
    big.inputs["Detail"].default_value = 10
    L.new(tc.outputs["Object"], big.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*albedo, 1)
    ramp.color_ramp.elements[1].color = (*lighter, 1)
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.35, 0.75
    L.new(big.outputs["Fac"], ramp.inputs["Fac"])
    L.new(ramp.outputs["Color"], inp(bsdf, "Base Color"))
    # metal grains: ~1 mm voronoi cells, a few percent of them are bare FeNi
    vor = N.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 700
    L.new(tc.outputs["Object"], vor.inputs["Vector"])
    sep = N.new("ShaderNodeMath")
    sep.operation = "GREATER_THAN"
    sep.inputs[1].default_value = 0.94
    L.new(vor.outputs["Color"], sep.inputs[0]) if "Color" in vor.outputs else None
    metal = N.new("ShaderNodeMath")
    metal.operation = "MULTIPLY"
    metal.inputs[1].default_value = 0.85
    L.new(sep.outputs[0], metal.inputs[0])
    L.new(metal.outputs[0], inp(bsdf, "Metallic"))
    rough = N.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.92, 0.32
    L.new(sep.outputs[0], rough.inputs["Value"])
    L.new(rough.outputs["Result"], inp(bsdf, "Roughness"))
    # micro relief: clumps (5 mm) + grains (0.5 mm)
    n1 = N.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 180
    n1.inputs["Detail"].default_value = 6
    n2 = N.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 1800
    for n in (n1, n2):
        L.new(tc.outputs["Object"], n.inputs["Vector"])
    add = N.new("ShaderNodeMath")
    add.operation = "ADD"
    L.new(n1.outputs["Fac"], add.inputs[0])
    L.new(n2.outputs["Fac"], add.inputs[1])
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55
    bump.inputs["Distance"].default_value = 0.002
    L.new(add.outputs[0], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], inp(bsdf, "Normal"))
    inp(bsdf, "Specular IOR Level", "Specular").default_value = 0.35
    return mat


def mat_simple(name, color, rough=0.5, metal=0.0, coat=0.0, sss=0.0):
    mat = bpy.data.materials.new(name)
    bsdf, _ = principled(mat)
    inp(bsdf, "Base Color").default_value = (*color, 1)
    inp(bsdf, "Roughness").default_value = rough
    inp(bsdf, "Metallic").default_value = metal
    if coat:
        inp(bsdf, "Coat Weight", "Clearcoat").default_value = coat
    if sss:
        inp(bsdf, "Subsurface Weight", "Subsurface").default_value = sss
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (0.002, 0.002, 0.002)
    return mat


def mat_rock(name="rock"):
    mat = mat_regolith(name, albedo=(0.10, 0.097, 0.092), lighter=(0.22, 0.21, 0.20))
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    inp(bsdf, "Specular IOR Level", "Specular").default_value = 0.55
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------
# world: black vacuum sky, stars only for camera rays
# ---------------------------------------------------------------------
def make_world(stars):
    w = bpy.data.worlds.new("space")
    bpy.context.scene.world = w
    w.use_nodes = True
    N, L = w.node_tree.nodes, w.node_tree.links
    out = N.get("World Output")
    bg = N.get("Background")
    bg.inputs["Color"].default_value = (0, 0, 0, 1)
    bg.inputs["Strength"].default_value = 0.0
    if stars == "none":
        return
    tc = N.new("ShaderNodeTexCoord")
    vor = N.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 260
    L.new(tc.outputs["Generated"], vor.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.035, 0.0
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.0, 1.0
    L.new(vor.outputs["Distance"], mr.inputs["Value"])
    pw = N.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 6.0
    L.new(mr.outputs["Result"], pw.inputs[0])
    tw = N.new("ShaderNodeMath")                           # per-star brightness from the cell colour
    tw.operation = "MULTIPLY"
    L.new(pw.outputs[0], tw.inputs[0])
    sepc = N.new("ShaderNodeSeparateColor") if hasattr(bpy.types, "ShaderNodeSeparateColor") else N.new("ShaderNodeSeparateRGB")
    L.new(vor.outputs["Color"], sepc.inputs[0])
    L.new(sepc.outputs[0], tw.inputs[1])
    em = N.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = {"faint": 6.0, "cinematic": 40.0}[stars]
    L.new(tw.outputs[0], em.inputs["Strength"]) if False else None
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = {"faint": 60.0, "cinematic": 400.0}[stars]
    L.new(tw.outputs[0], mul.inputs[0])
    L.new(mul.outputs[0], em.inputs["Strength"])
    lp = N.new("ShaderNodeLightPath")
    mix = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L.new(bg.outputs["Background"], mix.inputs[1])
    L.new(em.outputs["Emission"], mix.inputs[2])
    L.new(mix.outputs["Shader"], out.inputs["Surface"])


def make_sun(direction, strength=SUN_IRRADIANCE):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = strength
    light.angle = math.radians(SUN_ANGLE_DEG)
    light.color = (1.0, 0.985, 0.96)
    ob = link(bpy.data.objects.new("Sun", light))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector(direction).normalized().to_track_quat("Z", "Y")
    return ob


# ---------------------------------------------------------------------
# the surface
# ---------------------------------------------------------------------
def build_surface(R, q):
    xs, Z = R["xs"], R["Z"]
    n = len(xs)
    X, Y = np.meshgrid(xs, xs)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    ground = mesh_obj("Ground", V, F, smooth=True)
    reg = mat_regolith()
    assign(ground, reg)
    subsurf(ground, render=2 if q["scale"] == 100 else 1)
    displace(ground, tex("clumps", "CLOUDS", noise_scale=0.035, noise_depth=4), 0.006, coords="GLOBAL")
    displace(ground, tex("grain", "CLOUDS", noise_scale=0.006, noise_depth=2), 0.0012, coords="GLOBAL")
    far = mesh_obj("Far", R["horizon_v"], R["horizon_f"], smooth=True)
    assign(far, reg)
    # boulders: irregular, faceted, partly buried (MuJoCo collides with the matching ellipsoids)
    rock = mat_rock()
    vtex = tex("rockshape", "VORONOI", noise_scale=0.07, distance_metric="DISTANCE")
    ntex = tex("rocknoise", "CLOUDS", noise_scale=0.03, noise_depth=4)
    proto = icosphere("BoulderProto", 4)
    for i, b in enumerate(R["boulders"]):
        x, y, z, sx, sy, sz, yaw = b
        ob = link(bpy.data.objects.new(f"Boulder{i}", proto.data.copy()))
        ob.location, ob.scale, ob.rotation_euler = (x, y, z), (sx, sy, sz), (0.2 * math.sin(i), 0.15 * math.cos(3 * i), yaw)
        displace(ob, vtex, -0.5, coords="GLOBAL")
        displace(ob, ntex, 0.18, coords="GLOBAL")
        assign(ob, rock)
    bpy.data.objects.remove(proto)
    # pebbles and grains scattered on the ground
    peb = icosphere("Pebble", 2)
    displace(peb, tex("pebshape", "VORONOI", noise_scale=0.6), -0.3, coords="LOCAL")
    assign(peb, rock)
    peb.location = (0, 0, -50)
    peb.hide_render = False
    for name, count, size in (("pebbles", q["pebbles"], 0.006), ("cobbles", q["pebbles"] // 25, 0.022)):
        ps = ground.modifiers.new(name, "PARTICLE_SYSTEM").particle_system.settings
        ps.type = "HAIR"
        ps.count = count
        ps.hair_length = 1.0                      # instance scale = size x hair length
        ps.render_type = "OBJECT"
        ps.instance_object = peb
        ps.particle_size = size
        ps.size_random = 0.9
        ps.use_rotations = True
        ps.rotation_factor_random = 1.0
        ps.phase_factor_random = 2.0
        ps.distribution = "RAND"
    ground.show_instancer_for_render = True
    return ground


# ---------------------------------------------------------------------
# the cube (your CAD)
# ---------------------------------------------------------------------
def build_cube(R):
    cube = link(bpy.data.objects.new("Cubli", None))
    cube.rotation_mode = "QUATERNION"
    V, F = read_stl(os.path.join(ASSETS, "frame.stl"))
    frame = mesh_obj("Frame", V, F, smooth=True, sharp_deg=32)
    frame.parent = cube
    assign(frame, mat_simple("PLA_white", (0.80, 0.80, 0.78), rough=0.42, sss=0.08))
    bev = frame.modifiers.new("wn", "WEIGHTED_NORMAL") if hasattr(bpy.types, "WeightedNormalModifier") else None
    V, F = read_stl(os.path.join(ASSETS, "wheel.stl"))
    wmat = mat_simple("PLA_orange", (0.85, 0.22, 0.02), rough=0.38, sss=0.05)
    wheels = []
    for k in range(3):
        mount = link(bpy.data.objects.new(f"WheelMount{k}", None))
        mount.parent = cube
        mount.location = Vector(R["wheel_pos"][k])
        mount.rotation_mode = "QUATERNION"
        mount.rotation_quaternion = Quaternion(R["wheel_quat"][k])
        wh = mesh_obj(f"Wheel{k}", V, F, smooth=True, sharp_deg=32)
        wh.parent = mount
        wh.rotation_mode = "XYZ"
        assign(wh, wmat)
        wheels.append(wh)
    return cube, wheels


# ---------------------------------------------------------------------
# deployable sensors (LiDAR mast + hatch, camera arm + pan/tilt), rebuilt from the take's rig description
# ---------------------------------------------------------------------
SENSOR_LOOK = {  # MuJoCo material name -> (rough, metal, coat): appearance only, sizes come from the sim
    "lidar": (0.45, 0.0, 0.0), "glass": (0.04, 0.0, 1.0), "oakd": (0.5, 0.0, 0.0), "alu": (0.3, 1.0, 0.0),
    "hatch": (0.42, 0.0, 0.0), "bay": (0.6, 0.0, 0.0)}


def _empty(name, parent, loc=(0, 0, 0), quat=(1, 0, 0, 0)):
    ob = link(bpy.data.objects.new(name, None))
    ob.parent = parent
    ob.rotation_mode = "QUATERNION"
    ob.location, ob.rotation_quaternion = Vector(loc), Quaternion(quat)
    return ob


def _primitive(name, gtype, size):
    bm = bmesh.new()
    if gtype == 6:                                          # box (half sizes)
        bmesh.ops.create_cube(bm, size=2.0)
        scale = size[:3]
    elif gtype == 5:                                        # cylinder: radius, half length (axis = local z)
        bmesh.ops.create_cone(bm, cap_ends=True, segments=40, radius1=1.0, radius2=1.0, depth=2.0)
        scale = (size[0], size[0], size[1])
    else:                                                   # sphere / anything else
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        scale = (size[0],) * 3
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    ob.scale = scale
    return ob


def build_sensors(R, cube):
    """Returns {joint name: (animated empty, joint spec)}. Takes recorded before the sensors existed: {}."""
    if "sensor_rig" not in R:
        return {}
    spec = json.loads(str(R["sensor_rig"]))
    nodes, joints, mats = {"cube": cube}, {}, {}
    for b in spec:
        base = _empty(f"S_{b['name']}", nodes[b["parent"]], b["pos"], b["quat"])
        node = base
        if b["joint"]:
            jp = Vector(b["joint"]["pos"])
            mov = _empty(f"J_{b['name']}", _empty(f"Jp_{b['name']}", base, jp))
            node = _empty(f"Jb_{b['name']}", mov, -jp)
            joints[b["joint"]["name"]] = (mov, b["joint"])
        for i, g in enumerate(b["geoms"]):
            ob = _primitive(f"G_{b['name']}{i}", g["type"], g["size"])
            ob.parent = node
            ob.rotation_mode = "QUATERNION"
            ob.location, ob.rotation_quaternion = Vector(g["pos"]), Quaternion(g["quat"])
            key = g["mat"] or b["name"]
            if key not in mats:
                rough, metal, coat = SENSOR_LOOK.get(g["mat"], (0.5, 0.0, 0.0))
                mats[key] = mat_simple(f"sensor_{key}", tuple(g["rgba"][:3]), rough=rough, metal=metal, coat=coat)
            assign(ob, mats[key])
        nodes[b["name"]] = node
    return joints


def pose_sensors(joints, names, values, frame=None):
    """Set (and with a frame, keyframe) each recorded joint. values are flat: a ball joint (the sensor gimbal) takes
    4 (its quaternion), slide / hinge take 1. Takes from before the gimbal have only 1-value joints."""
    k = 0
    for n in names:
        j = joints[n][1] if n in joints else None
        width = 4 if j is not None and j["type"] == "ball" else 1
        v = values[k:k + width]
        k += width
        if j is None:
            continue
        mov = joints[n][0]
        if j["type"] == "slide":
            mov.location = Vector(j["axis"]) * float(v[0])
            if frame is not None:
                mov.keyframe_insert("location", frame=frame)
        else:
            mov.rotation_quaternion = (Quaternion([float(x) for x in v]).normalized() if j["type"] == "ball"
                                       else Quaternion(Vector(j["axis"]), float(v[0])))
            if frame is not None:
                mov.keyframe_insert("rotation_quaternion", frame=frame)


def key_sensors(joints, names, values, frame):
    pose_sensors(joints, names, values, frame)


# ---------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------
def slerp_track(t, Q, tq):
    i = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    a = np.clip((tq - t[i]) / (t[i + 1] - t[i]), 0, 1)[:, None]
    q0, q1 = Q[i], Q[i + 1]
    dot = np.sum(q0 * q1, 1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    th = np.arccos(np.clip(dot, -1, 1))
    s = np.sin(th)
    w0 = np.where(s > 1e-6, np.sin((1 - a) * th) / np.maximum(s, 1e-9), 1 - a)
    w1 = np.where(s > 1e-6, np.sin(a * th) / np.maximum(s, 1e-9), a)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def lerp_track(t, P, tq):
    return np.stack([np.interp(tq, t, P[:, j]) for j in range(P.shape[1])], 1)


def smooth(P, t, sigma):
    dt = np.median(np.diff(t))
    k = int(3 * sigma / dt)
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * dt / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(P, ((k, k), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], w, mode="valid") for j in range(P.shape[1])], 1)


def ground_h(R, x, y):
    xs, Z = R["xs"], R["Z"]
    j = np.clip(np.interp(x, xs, np.arange(len(xs))), 0, len(xs) - 1.001)
    i = np.clip(np.interp(y, xs, np.arange(len(xs))), 0, len(xs) - 1.001)
    i0, j0 = int(i), int(j)
    return float(Z[i0:i0 + 2, j0:j0 + 2].mean()) if abs(x) < xs[-1] and abs(y) < xs[-1] else float(Z.mean())


SHOTS = {  # distance m, elevation deg, focal mm, f-stop, orbit deg per video second, height floor
    "wide": (2.4, 5.0, 50, 16.0, 1.5),
    "cine": (0.72, 10.0, 40, 11.0, 4.0),
    "low": (0.42, 2.5, 24, 8.0, -3.0),
}


def make_cameras(R, a, frames, tvid, tsim, sim_smooth, cube):
    """One camera per shot; timeline markers cut between them (no motion-blur smear at cuts)."""
    sc = bpy.context.scene
    plan = []
    if a.cam == "auto" and a.still is None:
        seq = ["wide", "cine", "low", "cine", "wide", "low"]
        f, i = 0, 0
        while f < frames:
            n = int(a.fps * (8.0 if seq[i % len(seq)] != "wide" else 6.5))
            plan.append((seq[i % len(seq)], f, min(f + n, frames), 35.0 + 70.0 * i))
            f += n
            i += 1
    else:
        plan.append(("cine" if a.cam == "auto" else a.cam, 0, frames, 35.0))
    for si, (kind, f0, f1, az0) in enumerate(plan):
        dist, el, focal, fstop, orbit = SHOTS[kind]
        cd = bpy.data.cameras.new(f"Cam{si}_{kind}")
        cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = focal, 36.0, 0.005, 3000.0
        cd.dof.use_dof, cd.dof.focus_object, cd.dof.aperture_fstop = True, cube, fstop
        cam = link(bpy.data.objects.new(cd.name, cd))
        cam.rotation_mode = "QUATERNION"
        for f in range(max(f0 - 1, 0), min(f1 + 1, frames)):
            tgt = Vector(sim_smooth[f]) + Vector((0, 0, 0.035))
            az = math.radians(az0 + orbit * (tvid[f] - tvid[f0]))
            e = math.radians(el)
            pos = tgt + dist * Vector((math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)))
            pos.z = max(pos.z, ground_h(R, pos.x, pos.y) + 0.025)
            cam.location = pos
            cam.rotation_quaternion = (tgt - pos).to_track_quat("-Z", "Y")
            cam.keyframe_insert("location", frame=f + 1)
            cam.keyframe_insert("rotation_quaternion", frame=f + 1)
        mk = sc.timeline_markers.new(f"shot{si}", frame=f0 + 1)
        mk.camera = cam
        if si == 0:
            sc.camera = cam


def make_dust(R, a, tsim0, grain_mat):
    """Ejecta at every hard ground contact: ballistic, no drag, falls back under Psyche gravity."""
    sc = bpy.context.scene
    g = Vector(R["gravity"])
    sc.gravity = g
    if hasattr(sc, "use_gravity"):
        sc.use_gravity = True
    grain = icosphere("Grain", 1)
    assign(grain, grain_mat)
    grain.location = (0, 0, -60)
    sim_per_frame = a.speed / a.fps
    made = 0
    for ev in R["events"]:
        t, px, py, pz, nx, ny, nz, v = ev
        if v < 0.035:
            continue
        frame = int(round((t - tsim0) / sim_per_frame)) + 1
        if frame < 1 or frame > sc.frame_end:
            continue
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, radius=0.035, segments=12)
        me = bpy.data.meshes.new(f"DustEm{made}")
        bm.to_mesh(me)
        bm.free()
        em = link(bpy.data.objects.new(me.name, me))
        n = Vector((nx, ny, nz)).normalized()
        em.location = Vector((px, py, pz)) + 0.003 * n
        em.rotation_mode = "QUATERNION"
        em.rotation_quaternion = n.to_track_quat("Z", "Y")
        em.show_instancer_for_render = False
        ps = em.modifiers.new("dust", "PARTICLE_SYSTEM").particle_system.settings
        vn = max(0.5 * v, 0.02)
        ps.count = int(np.clip(v * 25000, 300, 4000))
        ps.frame_start, ps.frame_end = frame, frame + 1
        air = 2 * vn / g.length
        ps.lifetime = max(int(air / sim_per_frame * 1.15), 2)
        ps.lifetime_random = 0.6
        ps.emit_from = "FACE"
        ps.normal_factor = vn
        ps.factor_random = 0.6 * vn
        ps.physics_type = "NEWTON"
        ps.timestep = sim_per_frame
        ps.subframes = 2
        ps.render_type = "OBJECT"
        ps.instance_object = grain
        ps.particle_size = 0.0011
        ps.size_random = 0.85
        ps.use_rotations = True
        ps.angular_velocity_mode = "RAND"
        ps.angular_velocity_factor = 20
        ps.effector_weights.gravity = 1.0
        made += 1
    print(f"dust bursts: {made}", flush=True)


# ---------------------------------------------------------------------
# the asteroid from orbit (intro shot)
# ---------------------------------------------------------------------
NAMED = [  # lat, lon, diameter km, depth km  (Shepard et al. 2021 features; positions approximate)
    (-65, 260, 65, 4.0), (90, 0, 50, 2.5), (40, 300, 85, 2.0), (10, 105, 90, 3.0), (0, 270, 70, 3.0)]


def build_globe(R, q):
    a, b, c = (k / 2 for k in R["psyche_axes_km"])          # 1 Blender unit = 1 km here
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=384, v_segments=192, radius=1.0)
    me = bpy.data.meshes.new("Psyche")
    bm.to_mesh(me)
    bm.free()
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    u = co.reshape(-1, 3)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u / np.sqrt((u[:, 0] / a) ** 2 + (u[:, 1] / b) ** 2 + (u[:, 2] / c) ** 2)[:, None]
    r = np.linalg.norm(P, axis=1)
    rng = np.random.default_rng(5)
    craters = [(la, lo, d, dep) for la, lo, d, dep in NAMED]
    for _ in range(140):                                    # smaller craters, N(>D) ~ D^-2
        d = 6 * (1 - rng.random()) ** -0.5
        if d < 60:
            craters.append((math.degrees(math.asin(rng.uniform(-1, 1))), rng.uniform(0, 360), d, d * rng.uniform(0.05, 0.12)))
    for la, lo, d, dep in craters:
        cu = np.array([math.cos(math.radians(la)) * math.cos(math.radians(lo)),
                       math.cos(math.radians(la)) * math.sin(math.radians(lo)), math.sin(math.radians(la))])
        ang = np.arccos(np.clip(u @ cu, -1, 1)) * r / (d / 2)
        r += np.where(ang < 1, dep * (ang ** 2 - 1), 0) + 0.15 * dep * np.exp(-((ang - 1) / 0.22) ** 2)
    P = u * r[:, None]
    me.vertices.foreach_set("co", P.ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.update()
    ob = link(bpy.data.objects.new("Psyche", me))
    displace(ob, tex("globenoise", "CLOUDS", noise_scale=12.0, noise_depth=6), 1.6, coords="LOCAL")
    mat = mat_regolith("psyche_far", albedo=(0.10, 0.095, 0.09), lighter=(0.18, 0.17, 0.16))
    nodes = mat.node_tree.nodes
    for n in nodes:                                         # rescale the texture for km units
        if n.bl_idname == "ShaderNodeTexNoise":
            n.inputs["Scale"].default_value *= 0.05
        if n.bl_idname == "ShaderNodeBump":
            n.inputs["Distance"].default_value = 0.3
    assign(ob, mat)
    return ob, np.array(R["site_latlon"])


def render_intro(R, a, q, frames_dir):
    clear()
    setup_render(a, q)
    sc = bpy.context.scene
    sc.render.use_motion_blur = False                       # slow approach: blur adds nothing here
    make_world("cinematic" if a.stars != "none" else "none")
    globe, (lat, lon) = build_globe(R, q)
    site = Vector((math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                   math.cos(math.radians(lat)) * math.sin(math.radians(lon)), math.sin(math.radians(lat))))
    east = Vector((0, 0, 1)).cross(site)
    east = east.normalized() if east.length > 1e-3 else Vector((0, 1, 0))
    north = site.cross(east)
    sun_dir = (site * 0.42 + east * -0.85 + north * 0.3).normalized()  # low morning sun over the site
    make_sun(sun_dir, SUN_IRRADIANCE)
    ax, bx, cx = (k / 2 for k in R["psyche_axes_km"])
    r_site = 1.0 / math.sqrt((site.x / ax) ** 2 + (site.y / bx) ** 2 + (site.z / cx) ** 2)
    n = max(int(a.intro_seconds * a.fps), 2)
    cd = bpy.data.cameras.new("IntroCam")
    cd.lens, cd.clip_start, cd.clip_end = 50, 1.0, 10000
    cam = link(bpy.data.objects.new("IntroCam", cd))
    cam.rotation_mode = "QUATERNION"
    sc.camera = cam
    side = (east * -0.6 + north * 0.4).normalized()        # come in from the sunlit side
    for f in range(n):
        s = f / max(n - 1, 1)
        e = s * s * (3 - 2 * s)
        d = 950 * (1 - e) + (r_site + 300) * e
        pos = site * d + side * 160 * (1 - e)
        cam.location = pos
        look = site * r_site * 0.55 * e
        cam.rotation_quaternion = (look - pos).to_track_quat("-Z", "Y")
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_quaternion", frame=f + 1)
    sc.frame_start, sc.frame_end = 1, n
    sc.render.filepath = os.path.join(frames_dir, "a_intro_")
    bpy.ops.render.render(animation=True)
    print(f"intro: {n} frames")


# ---------------------------------------------------------------------
# render settings
# ---------------------------------------------------------------------
def setup_render(a, q):
    sc = bpy.context.scene
    r = sc.render
    w, h = (int(x) for x in a.res.lower().split("x"))
    r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, q["scale"]
    r.fps = a.fps
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.use_motion_blur = True
    if hasattr(r, "motion_blur_shutter"):
        r.motion_blur_shutter = 0.5
    sc.unit_settings.system = "METRIC"
    vs = sc.view_settings
    try:
        vs.view_transform = "AgX"
        vs.look = "AgX - Punchy"
    except TypeError:
        try:
            vs.view_transform = "Filmic"
            vs.look = "Filmic - High Contrast"
        except TypeError:
            pass
    vs.exposure = EXPOSURE
    if a.engine == "eevee":
        for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                r.engine = eng
                break
            except TypeError:
                continue
        return
    r.engine = "CYCLES"
    cy = sc.cycles
    cy.samples = a.samples or q["samples"]
    cy.use_denoising = True
    cy.max_bounces = q["bounces"]
    cy.diffuse_bounces = min(q["bounces"], 4)
    cy.glossy_bounces = min(q["bounces"], 4)
    cy.transparent_max_bounces = 8
    cy.use_adaptive_sampling = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = dev
                prefs.get_devices()
                if any(d.type == dev for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    cy.device = "GPU"
                    print(f"Cycles on GPU ({dev})")
                    break
            except TypeError:
                continue
    except Exception as e:                                  # noqa: BLE001
        print("GPU setup skipped:", e)


# ---------------------------------------------------------------------
def main():
    a = parse()
    q = QUALITY[a.quality]
    Rz = np.load(a.rec, allow_pickle=False)
    R = {k: Rz[k] for k in Rz.files}
    frames_dir = os.path.join(a.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    if a.still is None:                                     # start clean so old frames never mix in
        for f in os.listdir(frames_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(frames_dir, f))
    t = R["t"]
    t0 = a.start if a.start is not None else t[0]
    t1 = a.end if a.end is not None else t[-1]
    if a.still is not None:                                 # 3 frames around the moment, so motion blur works
        t0, t1 = a.still - a.speed / a.fps, a.still + 2 * a.speed / a.fps
    nfr = max(int(round((t1 - t0) / a.speed * a.fps)), 1)
    tvid = np.arange(nfr) / a.fps
    tsim = t0 + tvid * a.speed
    print(f"{R['gravity_name']}: {nfr} frames ({nfr / a.fps:.1f} s video, {t1 - t0:.1f} s sim, {a.speed}x)")
    if a.intro and a.still is None:
        render_intro(R, a, q, frames_dir)

    clear()
    setup_render(a, q)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, nfr
    make_world(a.stars)
    make_sun(R["sun"])
    build_surface(R, q)
    cube, wheels = build_cube(R)
    sensor_joints = build_sensors(R, cube)
    pos = lerp_track(t, R["pos"], tsim)
    quat = slerp_track(t, R["quat"], tsim)
    wang = lerp_track(t, R["wheel"], tsim)
    sq = lerp_track(t, R["sensor_q"], tsim) if sensor_joints else None
    # motion blur needs the pose half a frame either side: key every frame, linear
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:                                       # noqa: BLE001
        pass
    prev = None
    for f in range(nfr):
        qf = quat[f]
        if prev is not None and np.dot(prev, qf) < 0:
            qf = -qf
        prev = qf
        cube.location = Vector(pos[f])
        cube.rotation_quaternion = Quaternion(qf)
        cube.keyframe_insert("location", frame=f + 1)
        cube.keyframe_insert("rotation_quaternion", frame=f + 1)
        for k, wh in enumerate(wheels):
            wh.rotation_euler = (0, 0, float(wang[f, k]))
            wh.keyframe_insert("rotation_euler", index=2, frame=f + 1)
        if sensor_joints:                                   # mast, hatch, arm, pan, tilt from the recorded joints
            key_sensors(sensor_joints, list(R["sensor_joints"]), sq[f], f + 1)
    sim_smooth = lerp_track(t, smooth(R["pos"], t, 0.9), tsim)
    make_cameras(R, a, nfr, tvid, tsim, sim_smooth, cube)
    if a.still is None:
        make_dust(R, a, t0, bpy.data.materials["rock"])
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, "scene.blend"))
    if a.still is not None:
        sc.frame_set(2)
        sc.render.filepath = os.path.join(a.out, f"still_t{a.still:.1f}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath)
    elif a.only_frame is not None:
        sc.frame_set(a.only_frame)
        sc.render.filepath = os.path.join(a.out, f"frame_{a.only_frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath)
    else:
        sc.render.filepath = os.path.join(frames_dir, "b_main_")
        bpy.ops.render.render(animation=True)
        print(f"frames in {frames_dir}  ->  python psyche_cubli_v2.py video --frames \"{frames_dir}\"")


if __name__ == "__main__":
    main()
