"""
Livox Mid-360 -- live 2D/3D point-cloud viewer.

Connects to a Livox Mid-360 over Ethernet and renders its point stream in
real time with pygame: a fixed top-down 2D view, or a freely-rotatable 3D
view. This is a standalone visualization/inspection tool -- it does not talk
to the psyche-objdetec YOLO/terrain-risk pipeline, and it is unrelated to
this repo's separate FAST-LIO/RViz SLAM mapping pipeline ("mapping"); nothing
here should be merged with either.

Controls:
    F1 / on-screen "2D" button -- top-down view
    F2 / on-screen "3D" button -- free-rotate point cloud view
    Left-drag (3D only)        -- yaw + pitch rotate (0.12 deg/pixel)
    Scroll (3D only)           -- zoom
    Q                          -- quit

Run: python3 scripts/lidar_live_map.py
"""
import math
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import pygame

# ── Sensor / network config ──────────────────────────────────────────────
HOST_IP = "192.168.1.5"
POINT_PORT = 56301
HEADER_FMT = "<BHHHHBBB12sIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
POINT_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("r", "u1"), ("t", "u1")])
POINT_SIZE = 14

# ── Display / window layout ──────────────────────────────────────────────
W, H = 1280, 960
PANEL_H = 52
VH = H - PANEL_H             # viewport height (window minus bottom control panel)
CX, CY = W // 2, VH // 2

# ── Point lifetime + jitter suppression ──────────────────────────────────
EXPIRE_S = 0.1                # how long a point stays on screen before expiring
CELL_MM = 35                  # grid-cell size used to snap/dedupe points

# ── 2D view ───────────────────────────────────────────────────────────────
R2D_PX = min(W, VH) // 2 - 70
SCALE_2D = R2D_PX / 2000.0    # pixels per mm, fixed 2m radius view
RANGE_2D_MM = 2000.0

# ── 3D view ───────────────────────────────────────────────────────────────
MOUSE_DEG_PER_PX = 0.12       # yaw/pitch rotation sensitivity
ZOOM_STEP = 1.1
ZOOM_MIN, ZOOM_MAX = 0.005, 2.0
PERSPECTIVE_SCALE = 850.0
CAMERA_OFFSET_MM = 5000.0     # pulls the camera back from the origin
COLOR_RANGE_MM = 6000.0       # distance at which 3D points reach full "far" green


# ── UDP receiver thread ──────────────────────────────────────────────────
# Runs independently of the render loop so a slow frame never causes dropped
# UDP packets. Shares `scans` (a deque of (timestamp, x_mm, y_mm, z_mm) per
# packet) with the main thread under `scan_lock`.
scans = deque()
scan_lock = threading.Lock()
packet_count = 0
running = True


def udp_recv():
    global packet_count, running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    sock.bind((HOST_IP, POINT_PORT))
    sock.settimeout(1.0)

    while running:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError as e:
            if running:
                print(f"[lidar_live_map] UDP recv error: {e}")
            continue

        if len(data) < HEADER_SIZE:
            continue
        try:
            hdr = struct.unpack_from(HEADER_FMT, data, 0)
            n = min(hdr[3], len(data[HEADER_SIZE:]) // POINT_SIZE)
        except struct.error:
            continue
        if n == 0:
            continue

        pts = np.frombuffer(data[HEADER_SIZE:HEADER_SIZE + n * POINT_SIZE], dtype=POINT_DTYPE)
        x = pts["x"].astype(np.float32)
        y = pts["y"].astype(np.float32)
        z = pts["z"].astype(np.float32)
        valid = ~((x == 0) & (y == 0) & (z == 0))
        x, y, z = x[valid], y[valid], z[valid]
        if x.size == 0:
            continue

        with scan_lock:
            scans.append((time.monotonic(), x, y, z))
            packet_count += 1

    sock.close()


# ── Point processing ──────────────────────────────────────────────────────
def collect_live_points(now):
    """Drops expired scans, then returns all remaining points concatenated
    into flat (x_mm, y_mm, z_mm, age_frac) arrays. age_frac is 1.0 for a
    point that just arrived and fades to 0.0 as it approaches EXPIRE_S,
    used to fade older points out smoothly instead of having them pop off
    screen the instant they expire."""
    with scan_lock:
        while scans and now - scans[0][0] > EXPIRE_S:
            scans.popleft()
        snapshot = list(scans)

    if not snapshot:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty

    xs, ys, zs, ages = [], [], [], []
    for ts, x, y, z in snapshot:
        xs.append(x)
        ys.append(y)
        zs.append(z)
        age_frac = 1.0 - (now - ts) / EXPIRE_S
        ages.append(np.full(x.shape, age_frac, dtype=np.float32))

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(ages)


def declutter(x_mm, y_mm, z_mm, age):
    """Snaps points to a CELL_MM grid and keeps one point per occupied cell.

    This is the fix for visual jitter on static objects: raw sensor noise
    means a stationary surface's returns wobble by a few mm between scans,
    which reads as flicker once drawn. Snapping to a fixed grid means the
    same physical spot always quantizes to the same on-screen position, so
    it stops moving -- and it collapses the many duplicate returns a static
    surface accumulates over the EXPIRE_S window into a single dot, which
    also keeps per-frame point counts (and render time) down.
    """
    if x_mm.size == 0:
        return x_mm, y_mm, z_mm, age

    grid = np.stack(
        [
            np.round(x_mm / CELL_MM).astype(np.int64),
            np.round(y_mm / CELL_MM).astype(np.int64),
            np.round(z_mm / CELL_MM).astype(np.int64),
        ],
        axis=1,
    )
    # Keep the freshest sample in each cell so age-based fade stays accurate.
    order = np.argsort(-age, kind="stable")
    grid, age = grid[order], age[order]
    _, first_idx = np.unique(grid, axis=0, return_index=True)

    return (
        grid[first_idx, 0].astype(np.float32) * CELL_MM,
        grid[first_idx, 1].astype(np.float32) * CELL_MM,
        grid[first_idx, 2].astype(np.float32) * CELL_MM,
        age[first_idx],
    )


# ── 3D projection ──────────────────────────────────────────────────────────
def project3d(x_mm, y_mm, z_mm, yaw_r, pitch_r, zoom, cx, cy):
    """Yaw + pitch rotate about the sensor origin, then apply a simple
    perspective projection. Vectorized (numpy arrays in, arrays out)."""
    x1 = x_mm * np.cos(yaw_r) + y_mm * np.sin(yaw_r)
    y1 = -x_mm * np.sin(yaw_r) + y_mm * np.cos(yaw_r)
    y2 = y1 * np.cos(pitch_r) + z_mm * np.sin(pitch_r)
    z2 = -y1 * np.sin(pitch_r) + z_mm * np.cos(pitch_r)

    depth = np.maximum(y2 + CAMERA_OFFSET_MM, 50.0)
    scale = PERSPECTIVE_SCALE * zoom / depth
    sx = (cx + x1 * scale).astype(np.int32)
    sy = (cy - z2 * scale).astype(np.int32)
    return sx, sy, depth


def project3d_single(x, y, z, yaw_r, pitch_r, zoom, cx, cy):
    """Scalar version of project3d, for one-off points (grid lines, rings,
    markers) where building a length-1 numpy array would just be overhead."""
    x1 = x * math.cos(yaw_r) + y * math.sin(yaw_r)
    y1 = -x * math.sin(yaw_r) + y * math.cos(yaw_r)
    y2 = y1 * math.cos(pitch_r) + z * math.sin(pitch_r)
    z2 = -y1 * math.sin(pitch_r) + z * math.cos(pitch_r)

    depth = max(y2 + CAMERA_OFFSET_MM, 50.0)
    scale = PERSPECTIVE_SCALE * zoom / depth
    return int(cx + x1 * scale), int(cy - z2 * scale)


# ── Drawing helpers ─────────────────────────────────────────────────────────
def draw_button(screen, font, label, rect, active):
    pygame.draw.rect(screen, (0, 110, 210) if active else (215, 215, 215), rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), rect, 1, border_radius=6)
    text = font.render(label, True, (255, 255, 255) if active else (50, 50, 50))
    screen.blit(text, (rect[0] + (rect[2] - text.get_width()) // 2,
                        rect[1] + (rect[3] - text.get_height()) // 2))


def draw_2d(screen, font_b, font, font_s, x_mm, y_mm, age, pkt_count, fps):
    screen.fill((255, 255, 255))

    for r_m, label in [(0.5, "0.5m"), (1, "1m"), (1.5, "1.5m"), (2, "2m")]:
        r_px = int(r_m * 1000 * SCALE_2D)
        ring_color = (170, 170, 170) if r_m < 2 else (90, 90, 90)
        pygame.draw.circle(screen, ring_color, (CX, CY), r_px, 1)
        screen.blit(font_s.render(label, True, (120, 120, 120)), (CX + r_px + 3, CY - 9))
    pygame.draw.line(screen, (220, 220, 220), (0, CY), (W, CY), 1)
    pygame.draw.line(screen, (220, 220, 220), (CX, 0), (CX, VH), 1)

    dist = np.sqrt(x_mm ** 2 + y_mm ** 2)
    px = (CX + x_mm * SCALE_2D).astype(np.int32)
    py = (CY - y_mm * SCALE_2D).astype(np.int32)
    in_view = (px >= 0) & (px < W) & (py >= 0) & (py < VH) & (dist <= RANGE_2D_MM)
    px, py, dist, age = px[in_view], py[in_view], dist[in_view], age[in_view]

    # Red = close, blue = far; fade toward the background as points age out.
    closeness = np.clip(1.0 - dist / RANGE_2D_MM, 0, 1)
    r = np.clip(closeness * 255 * age, 0, 255).astype(np.int32)
    g = np.clip(closeness * (1 - closeness) * 320 * age, 0, 255).astype(np.int32)
    b = np.clip((1 - closeness) * 220 * age, 0, 255).astype(np.int32)
    for i in range(len(px)):
        pygame.draw.circle(screen, (int(r[i]), int(g[i]), int(b[i])), (px[i], py[i]), 2)

    # Sensor marker + forward-facing indicator.
    pygame.draw.circle(screen, (200, 0, 0), (CX, CY), 9)
    pygame.draw.line(screen, (0, 160, 0), (CX, CY), (CX, CY - 34), 3)
    pygame.draw.polygon(screen, (0, 160, 0), [(CX, CY - 46), (CX - 6, CY - 34), (CX + 6, CY - 34)])
    screen.blit(font_s.render("FWD", True, (0, 130, 0)), (CX + 11, CY - 44))

    screen.blit(font_b.render("TOP-DOWN  2m MAP", True, (0, 0, 110)), (12, 8))
    screen.blit(font.render(f"Pkts:{pkt_count:,}  Pts:{len(px):,}  FPS:{fps:.0f}", True, (0, 0, 160)), (12, 34))


def draw_3d(screen, font_b, font, font_s, x_mm, y_mm, z_mm, yaw, pitch, zoom, pkt_count, fps):
    screen.fill((8, 10, 20))
    yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)

    def proj1(px, py, pz):
        return project3d_single(px, py, pz, yaw_r, pitch_r, zoom, CX, CY)

    # Ground grid, drawn at an approximate floor height below the sensor.
    floor_z = -1800.0
    grid_color = (0, 40, 80)
    for gv in range(-6000, 7000, 1000):
        line = [proj1(gv, -6000, floor_z), proj1(gv, 6000, floor_z)]
        if all(-200 < p[0] < W + 200 and -200 < p[1] < H + 200 for p in line):
            pygame.draw.line(screen, grid_color, line[0], line[1], 1)
        line = [proj1(-6000, gv, floor_z), proj1(6000, gv, floor_z)]
        if all(-200 < p[0] < W + 200 and -200 < p[1] < H + 200 for p in line):
            pygame.draw.line(screen, grid_color, line[0], line[1], 1)

    # Ground-plane distance rings.
    for r_mm in (1000, 2000, 3000, 5000, 8000):
        ring_pts = [
            proj1(r_mm * math.cos(math.radians(deg)), r_mm * math.sin(math.radians(deg)), floor_z)
            for deg in range(0, 361, 12)
        ]
        if any(-100 < p[0] < W + 100 and -100 < p[1] < H + 100 for p in ring_pts):
            try:
                pygame.draw.lines(screen, (0, 70, 130), False, ring_pts, 1)
            except ValueError:
                pass  # off-screen degenerate polygon; skip this ring this frame

    # Point cloud.
    total_drawn = 0
    if x_mm.size:
        sx, sy, depth = project3d(x_mm, y_mm, z_mm, yaw_r, pitch_r, zoom, CX, CY)
        in_view = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < VH)
        sx, sy, depth = sx[in_view], sy[in_view], depth[in_view]
        dist_3d = np.sqrt(x_mm[in_view] ** 2 + y_mm[in_view] ** 2 + z_mm[in_view] ** 2)
        total_drawn = len(sx)

        # Back-to-front so nearer points draw over farther ones.
        order = np.argsort(-depth)
        sx, sy, dist_3d = sx[order], sy[order], dist_3d[order]

        # Red = close, green = far.
        closeness = np.clip(1.0 - dist_3d / COLOR_RANGE_MM, 0, 1)
        red = np.clip(closeness * 255, 0, 255).astype(np.int32)
        green = np.clip((1 - closeness) * 255, 0, 255).astype(np.int32)
        for i in range(len(sx)):
            pygame.draw.circle(screen, (int(red[i]), int(green[i]), 0), (sx[i], sy[i]), 2)

    # Sensor marker + forward-facing indicator.
    ox, oy = proj1(0, 0, 0)
    pygame.draw.circle(screen, (255, 60, 60), (ox, oy), 10)
    pygame.draw.circle(screen, (255, 180, 180), (ox, oy), 10, 2)
    screen.blit(font_s.render("YOU", True, (255, 100, 100)), (ox + 12, oy - 8))
    fx, fy = proj1(0, 1500, 0)
    pygame.draw.line(screen, (0, 255, 120), (ox, oy), (fx, fy), 3)
    screen.blit(font_s.render("FWD", True, (0, 220, 100)), (fx + 4, fy - 8))

    screen.blit(font_b.render("3D POINT CLOUD", True, (0, 220, 255)), (12, 8))
    screen.blit(font.render(f"Pkts:{pkt_count:,}  Pts:{total_drawn:,}  FPS:{fps:.0f}", True, (0, 180, 230)), (12, 34))
    screen.blit(
        font_s.render(
            f"Yaw:{yaw % 360:.0f}deg  Pitch:{pitch:.0f}deg  Drag=rotate  Scroll=zoom  RED=close  GREEN=far",
            True, (0, 100, 160),
        ),
        (12, 56),
    )


def main():
    global running

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Livox Mid-360 -- Live Map")
    font_bold = pygame.font.SysFont("monospace", 20, bold=True)
    font = pygame.font.SysFont("monospace", 15)
    font_small = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()

    threading.Thread(target=udp_recv, daemon=True).start()

    mode = "2D"
    yaw, pitch, zoom = 0.0, 30.0, 0.045
    dragging = False
    last_mouse_pos = (0, 0)

    btn_w, btn_h = 110, 34
    btn_2d = pygame.Rect(W // 2 - btn_w - 8, H - PANEL_H + 9, btn_w, btn_h)
    btn_3d = pygame.Rect(W // 2 + 8, H - PANEL_H + 9, btn_w, btn_h)

    while running:
        clock.tick(30)
        now = time.monotonic()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_F1:
                    mode = "2D"
                elif ev.key == pygame.K_F2:
                    mode = "3D"
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if btn_2d.collidepoint(ev.pos):
                    mode = "2D"
                elif btn_3d.collidepoint(ev.pos):
                    mode = "3D"
                elif ev.button == 1:
                    dragging = True
                    last_mouse_pos = ev.pos
                elif ev.button == 4 and mode == "3D":
                    zoom = min(zoom * ZOOM_STEP, ZOOM_MAX)
                elif ev.button == 5 and mode == "3D":
                    zoom = max(zoom / ZOOM_STEP, ZOOM_MIN)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    dragging = False
            elif ev.type == pygame.MOUSEMOTION:
                if dragging and mode == "3D":
                    dx, dy = ev.pos[0] - last_mouse_pos[0], ev.pos[1] - last_mouse_pos[1]
                    yaw += dx * MOUSE_DEG_PER_PX
                    pitch = max(-89.0, min(89.0, pitch + dy * MOUSE_DEG_PER_PX))
                    last_mouse_pos = ev.pos

        x_mm, y_mm, z_mm, age = declutter(*collect_live_points(now))
        with scan_lock:
            pkt_count = packet_count

        if mode == "2D":
            draw_2d(screen, font_bold, font, font_small, x_mm, y_mm, age, pkt_count, clock.get_fps())
        else:
            draw_3d(screen, font_bold, font, font_small, x_mm, y_mm, z_mm, yaw, pitch, zoom, pkt_count, clock.get_fps())

        # Bottom control panel.
        pygame.draw.rect(screen, (232, 232, 232), (0, VH, W, PANEL_H))
        pygame.draw.line(screen, (180, 180, 180), (0, VH), (W, VH), 1)
        draw_button(screen, font, "  2D  ", btn_2d, mode == "2D")
        draw_button(screen, font, "  3D  ", btn_3d, mode == "3D")
        screen.blit(font_small.render("F1=2D  F2=3D  Q=quit", True, (130, 130, 130)), (12, VH + 18))

        pygame.display.flip()

    running = False
    pygame.quit()


if __name__ == "__main__":
    main()
