"""
Livox Mid-360 — 2D + 3D Live Map
2D: top-down fixed 2m, super fast 0.06s points
3D: Waymo-style, locked pitch, only left/right rotate
Run: python3 lidar_map.py
"""
import socket, struct, threading, time, math
import numpy as np
import pygame
from collections import deque

HOST_IP    = "192.168.1.5"
POINT_PORT = 56301
HEADER_FMT = '<BHHHHBBB12sIQ'
HEADER_SIZE= struct.calcsize(HEADER_FMT)
POINT_DTYPE= np.dtype([('x','<i4'),('y','<i4'),('z','<i4'),('r','u1'),('t','u1')])
POINT_SIZE = 14

W, H      = 1280, 960
EXPIRE_2D = 0.06   # seconds — super snappy
EXPIRE_3D = 4.0    # seconds — dense 3D cloud
PITCH_DEG = 38.0   # fixed camera tilt, never changes
FLOOR_Z   = -1800  # mm below sensor (approx floor)
TOP_Z     =  600   # mm above sensor (approx ceiling)

# ── Height colour LUT ─────────────────────────────────────
def make_lut(n=256):
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t = i / (n-1)
        if   t < 0.2:  f=t/0.2;       lut[i]=[20,        20,           int(80+f*120)]
        elif t < 0.45: f=(t-0.2)/0.25; lut[i]=[0,         int(f*255),   200]
        elif t < 0.65: f=(t-0.45)/0.2; lut[i]=[0,         255,          int(200*(1-f))]
        elif t < 0.85: f=(t-0.65)/0.2; lut[i]=[int(f*255),255,          0]
        else:          f=(t-0.85)/0.15;lut[i]=[255,        int(255*(1-f)),0]
    return lut
LUT = make_lut()

def z_to_color(zm):
    t = np.clip((zm - FLOOR_Z) / (TOP_Z - FLOOR_Z), 0, 1)
    idx = (t * 255).astype(np.int32)
    return LUT[idx]

# ── Global scan buffer ────────────────────────────────────
scans   = deque()
scan_lk = threading.Lock()
pkt_cnt = 0
running = True

def udp_recv():
    global pkt_cnt, running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
    sock.bind((HOST_IP, POINT_PORT))
    sock.settimeout(1.0)
    while running:
        try:   data,_ = sock.recvfrom(65535)
        except socket.timeout: continue
        except Exception as e:
            if running: print(e)
            continue
        if len(data) < HEADER_SIZE: continue
        try:
            hdr = struct.unpack_from(HEADER_FMT, data, 0)
            n   = min(hdr[3], len(data[HEADER_SIZE:]) // POINT_SIZE)
        except: continue
        if n == 0: continue
        pts = np.frombuffer(data[HEADER_SIZE:HEADER_SIZE+n*POINT_SIZE], dtype=POINT_DTYPE)
        x=pts['x'].astype(np.float32)
        y=pts['y'].astype(np.float32)
        z=pts['z'].astype(np.float32)
        ok=~((x==0)&(y==0)&(z==0))
        x=x[ok]; y=y[ok]; z=z[ok]
        if len(x)==0: continue
        with scan_lk:
            scans.append((time.monotonic(), x, y, z))
            pkt_cnt += 1
    sock.close()

# ── 3D project: yaw-only, fixed pitch ────────────────────
def project3d(xm, ym, zm, yaw_r, zoom, cx, cy):
    pr = math.radians(PITCH_DEG)
    # Yaw (horizontal spin only)
    x1 =  xm * np.cos(yaw_r) + ym * np.sin(yaw_r)
    y1 = -xm * np.sin(yaw_r) + ym * np.cos(yaw_r)
    # Fixed pitch tilt
    y2 =  y1 * np.cos(pr) + zm * np.sin(pr)
    z2 = -y1 * np.sin(pr) + zm * np.cos(pr)
    # Perspective (y2=depth, z2=screen-vertical, x1=screen-horizontal)
    depth = y2 + 4500
    depth = np.where(depth < 50, 50, depth)
    s = 900.0 * zoom / depth
    sx = (cx + x1 * s).astype(np.int32)
    sy = (cy - z2 * s).astype(np.int32)
    return sx, sy, depth

def project3d_single(x, y, z, yaw_r, zoom, cx, cy):
    pr = math.radians(PITCH_DEG)
    x1 =  x*math.cos(yaw_r) + y*math.sin(yaw_r)
    y1 = -x*math.sin(yaw_r) + y*math.cos(yaw_r)
    y2 =  y1*math.cos(pr)   + z*math.sin(pr)
    z2 = -y1*math.sin(pr)   + z*math.cos(pr)
    depth = y2 + 4500
    if depth < 50: depth = 50
    s = 900.0 * zoom / depth
    return int(cx + x1*s), int(cy - z2*s)

def draw_btn(screen, font, label, rect, active):
    pygame.draw.rect(screen, (0,120,220) if active else (220,220,220), rect, border_radius=6)
    pygame.draw.rect(screen, (100,100,100), rect, 1, border_radius=6)
    t = font.render(label, True, (255,255,255) if active else (60,60,60))
    screen.blit(t, (rect[0]+(rect[2]-t.get_width())//2, rect[1]+(rect[3]-t.get_height())//2))

def main():
    global running
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Livox Mid-360  —  Live Map")
    font_b = pygame.font.SysFont('monospace', 20, bold=True)
    font   = pygame.font.SysFont('monospace', 15)
    font_s = pygame.font.SysFont('monospace', 12)
    clock  = pygame.time.Clock()

    threading.Thread(target=udp_recv, daemon=True).start()

    mode  = "2D"
    yaw   = 0.0
    zoom  = 0.048
    drag  = False
    lx_   = 0

    PANEL_H = 58
    VH      = H - PANEL_H  # viewport height
    CX, CY  = W//2, VH//2
    R2D_PX  = min(W, VH)//2 - 70
    SCALE2D = R2D_PX / 2000.0   # 2m radius

    BTN_W, BTN_H = 120, 36
    btn_2d = pygame.Rect(W//2 - BTN_W - 8, H-PANEL_H+11, BTN_W, BTN_H)
    btn_3d = pygame.Rect(W//2 + 8,          H-PANEL_H+11, BTN_W, BTN_H)

    while running:
        clock.tick(30)
        now = time.monotonic()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            elif ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_q:  running = False
                elif ev.key == pygame.K_F1: mode = "2D"
                elif ev.key == pygame.K_F2: mode = "3D"
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if btn_2d.collidepoint(ev.pos):   mode = "2D"
                elif btn_3d.collidepoint(ev.pos): mode = "3D"
                elif ev.button == 1 and mode == "3D": drag=True; lx_=ev.pos[0]
                elif ev.button == 4 and mode == "3D": zoom=min(zoom*1.12, 2.0)
                elif ev.button == 5 and mode == "3D": zoom=max(zoom/1.12, 0.008)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1: drag = False
            elif ev.type == pygame.MOUSEMOTION:
                if drag and mode == "3D":
                    yaw += (ev.pos[0] - lx_) * 0.35   # LEFT/RIGHT only
                    lx_  = ev.pos[0]

        expire = EXPIRE_2D if mode == "2D" else EXPIRE_3D
        with scan_lk:
            while scans and now - scans[0][0] > expire: scans.popleft()
            snap = list(scans)

        yr = math.radians(yaw)

        # ════════════════ 2D ════════════════════════════════
        if mode == "2D":
            screen.fill((255, 255, 255))
            # rings
            for r_m, lbl in [(0.5,"0.5m"),(1,"1m"),(1.5,"1.5m"),(2,"2m")]:
                r_px = int(r_m*1000*SCALE2D)
                pygame.draw.circle(screen,(160,160,160) if r_m<2 else (80,80,80),(CX,CY),r_px,1)
                screen.blit(font_s.render(lbl,True,(110,110,110)),(CX+r_px+3,CY-9))
            pygame.draw.line(screen,(220,220,220),(0,CY),(W,CY),1)
            pygame.draw.line(screen,(220,220,220),(CX,0),(CX,VH),1)

            total = 0
            for ts,xmm,ymm,zmm in snap:
                fade = 1.0 - (now-ts)/EXPIRE_2D
                px=(CX+xmm*SCALE2D).astype(np.int32)
                py=(CY-ymm*SCALE2D).astype(np.int32)
                dist=np.sqrt(xmm**2+ymm**2)
                ok=(px>=0)&(px<W)&(py>=0)&(py<VH)&(dist<=2000)
                px=px[ok]; py=py[ok]; dist=dist[ok]
                if len(px)==0: continue
                total+=len(px)
                t=np.clip(1.0-dist/2000.0,0,1)
                r=np.clip((t*230*fade).astype(int),0,255)
                g=np.clip((t*(1-t)*320*fade).astype(int),0,255)
                b=np.clip(((1-t)*210*fade).astype(int),0,255)
                for i in range(len(px)):
                    pygame.draw.circle(screen,(int(r[i]),int(g[i]),int(b[i])),(px[i],py[i]),2)

            # sensor
            pygame.draw.circle(screen,(200,0,0),(CX,CY),9)
            pygame.draw.line(screen,(0,160,0),(CX,CY),(CX,CY-32),3)
            pygame.draw.polygon(screen,(0,160,0),[(CX,CY-44),(CX-6,CY-32),(CX+6,CY-32)])
            screen.blit(font_s.render("FWD",True,(0,130,0)),(CX+10,CY-42))

            with scan_lk: pk=pkt_cnt
            screen.blit(font_b.render("TOP-DOWN  2m LIVE MAP",True,(0,0,110)),(12,8))
            screen.blit(font.render(f"Pkts:{pk:,}  Pts:{total:,}  FPS:{clock.get_fps():.0f}",True,(0,0,160)),(12,34))

        # ════════════════ 3D ════════════════════════════════
        else:
            screen.fill((6, 8, 16))

            # Ground grid at FLOOR_Z
            pr = math.radians(PITCH_DEG)
            grid_col = (0, 40, 80)
            for gv in range(-6000, 7000, 1000):
                # Lines along X
                pts_line = []
                for gx in (-6000, 6000):
                    sx,sy = project3d_single(float(gx),float(gv),float(FLOOR_Z),yr,zoom,CX,CY)
                    pts_line.append((sx,sy))
                if all(-200<p[0]<W+200 and -200<p[1]<H+200 for p in pts_line):
                    pygame.draw.line(screen, grid_col, pts_line[0], pts_line[1], 1)
                # Lines along Y
                pts_line = []
                for gy in (-6000, 6000):
                    sx,sy = project3d_single(float(gv),float(gy),float(FLOOR_Z),yr,zoom,CX,CY)
                    pts_line.append((sx,sy))
                if all(-200<p[0]<W+200 and -200<p[1]<H+200 for p in pts_line):
                    pygame.draw.line(screen, grid_col, pts_line[0], pts_line[1], 1)

            # Distance rings on ground
            for r_mm in [1000,2000,3000,5000,8000]:
                ring_pts=[]
                for deg in range(0,361,12):
                    rx=r_mm*math.cos(math.radians(deg))
                    ry=r_mm*math.sin(math.radians(deg))
                    sx,sy=project3d_single(rx,ry,float(FLOOR_Z),yr,zoom,CX,CY)
                    ring_pts.append((sx,sy))
                ring_in = [p for p in ring_pts if -100<p[0]<W+100 and -100<p[1]<H+100]
                if len(ring_in)>3:
                    try: pygame.draw.lines(screen,(0,70,130),False,ring_pts,1)
                    except: pass
                lbl=f"{r_mm//1000}m"
                lx_=int(CX+r_mm*math.cos(math.radians(-30-yaw%360))*zoom*900/4500)
                ly_=int(CY-FLOOR_Z*math.sin(math.radians(PITCH_DEG))*zoom*900/4500)
                sx0,sy0=project3d_single(r_mm*math.cos(math.radians(60)),
                                         r_mm*math.sin(math.radians(60)),
                                         float(FLOOR_Z),yr,zoom,CX,CY)
                if 0<sx0<W and 0<sy0<VH:
                    screen.blit(font_s.render(lbl,True,(0,100,180)),(sx0+3,sy0-10))

            # Collect points
            all_x=[]; all_y=[]; all_z=[]
            for ts,xmm,ymm,zmm in snap:
                all_x.append(xmm); all_y.append(ymm); all_z.append(zmm)

            total=0
            if all_x:
                xmm=np.concatenate(all_x)
                ymm=np.concatenate(all_y)
                zmm=np.concatenate(all_z)

                sx,sy,depth=project3d(xmm,ymm,zmm,yr,zoom,CX,CY)
                ok=(sx>=0)&(sx<W)&(sy>=0)&(sy<VH)
                sx=sx[ok]; sy=sy[ok]; depth=depth[ok]; zm_ok=zmm[ok]
                total=len(sx)

                # Back to front for correct overlap
                order=np.argsort(-depth)
                sx=sx[order]; sy=sy[order]; zm_ok=zm_ok[order]

                cols=z_to_color(zm_ok)
                for i in range(len(sx)):
                    c=cols[i]
                    pygame.draw.circle(screen,(int(c[0]),int(c[1]),int(c[2])),(sx[i],sy[i]),2)

            # YOU ARE HERE — sensor marker projected at origin
            ox,oy=project3d_single(0,0,0,yr,zoom,CX,CY)
            pygame.draw.circle(screen,(255,60,60),(ox,oy),10)
            pygame.draw.circle(screen,(255,180,180),(ox,oy),10,2)
            screen.blit(font_s.render("YOU",True,(255,100,100)),(ox+12,oy-8))

            # Forward indicator
            fx,fy=project3d_single(0,1500,0,yr,zoom,CX,CY)
            pygame.draw.line(screen,(0,255,120),(ox,oy),(fx,fy),3)
            screen.blit(font_s.render("FWD",True,(0,220,100)),(fx+4,fy-8))

            with scan_lk: pk=pkt_cnt
            screen.blit(font_b.render("3D POINT CLOUD  —  WAYMO VIEW",True,(0,220,255)),(12,8))
            screen.blit(font.render(f"Pkts:{pk:,}  Pts:{total:,}  FPS:{clock.get_fps():.0f}",True,(0,180,230)),(12,34))
            screen.blit(font_s.render("← Drag left/right to rotate  |  Scroll to zoom",True,(0,100,160)),(12,56))

            # Height legend
            lx_leg=W-115; ly_leg=70
            screen.blit(font_s.render("HEIGHT",True,(200,200,255)),(lx_leg,ly_leg)); ly_leg+=14
            for i in range(160):
                c=LUT[int(i/160*255)]
                pygame.draw.rect(screen,(int(c[0]),int(c[1]),int(c[2])),(lx_leg,ly_leg+i,14,1))
            screen.blit(font_s.render("HIGH",True,(255,150,0)),(lx_leg+18,ly_leg))
            screen.blit(font_s.render("MID", True,(0,255,100)),(lx_leg+18,ly_leg+80))
            screen.blit(font_s.render("LOW", True,(50,80,180)),(lx_leg+18,ly_leg+152))

        # ── Bottom panel ──────────────────────────────────
        pygame.draw.rect(screen,(235,235,235),(0,VH,W,PANEL_H))
        pygame.draw.line(screen,(180,180,180),(0,VH),(W,VH),1)
        draw_btn(screen,font,"  2D  ",btn_2d,mode=="2D")
        draw_btn(screen,font,"  3D  ",btn_3d,mode=="3D")
        screen.blit(font_s.render("F1=2D  F2=3D  Q=quit",True,(130,130,130)),(12,VH+20))

        pygame.display.flip()

    running=False
    pygame.quit()

if __name__=="__main__":
    main()
