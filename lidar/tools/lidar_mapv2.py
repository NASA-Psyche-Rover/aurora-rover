"""
Livox Mid-360 — 2D + 3D Live Map
2D: stable objects 0.5s, moving/fresh 0.05s, no jitter
3D: full rotation, red=close green=far, 0.1s, fast
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
PANEL_H   = 52
VH        = H - PANEL_H

# 2D constants
CX2, CY2  = W//2, VH//2
R2D_PX    = min(W,VH)//2 - 70
SCALE2D   = R2D_PX / 2000.0     # 2m radius
CELL_MM   = 35                   # grid cell size mm for stable map

# ── Shared scan ring buffer ───────────────────────────────
scans   = deque()       # (timestamp, x_mm, y_mm, z_mm)
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
        if len(data)<HEADER_SIZE: continue
        try:
            hdr = struct.unpack_from(HEADER_FMT,data,0)
            n   = min(hdr[3], len(data[HEADER_SIZE:])//POINT_SIZE)
        except: continue
        if n==0: continue
        pts = np.frombuffer(data[HEADER_SIZE:HEADER_SIZE+n*POINT_SIZE],dtype=POINT_DTYPE)
        x=pts['x'].astype(np.float32)
        y=pts['y'].astype(np.float32)
        z=pts['z'].astype(np.float32)
        ok=~((x==0)&(y==0)&(z==0))
        x=x[ok]; y=y[ok]; z=z[ok]
        if len(x)==0: continue
        with scan_lk:
            scans.append((time.monotonic(),x,y,z))
            pkt_cnt+=1
    sock.close()

# ── 3D projection ─────────────────────────────────────────
def project3d(xm,ym,zm,yaw_r,pitch_r,zoom,cx,cy):
    # Yaw
    x1 =  xm*np.cos(yaw_r) + ym*np.sin(yaw_r)
    y1 = -xm*np.sin(yaw_r) + ym*np.cos(yaw_r)
    # Pitch
    y2 =  y1*np.cos(pitch_r) + zm*np.sin(pitch_r)
    z2 = -y1*np.sin(pitch_r) + zm*np.cos(pitch_r)
    # Perspective
    depth = y2 + 5000
    depth = np.where(depth<50,50,depth)
    s = 850.0*zoom/depth
    sx=(cx+x1*s).astype(np.int32)
    sy=(cy-z2*s).astype(np.int32)
    return sx,sy,depth

def draw_btn(screen,font,label,rect,active):
    pygame.draw.rect(screen,(0,110,210) if active else (215,215,215),rect,border_radius=6)
    pygame.draw.rect(screen,(120,120,120),rect,1,border_radius=6)
    t=font.render(label,True,(255,255,255) if active else (50,50,50))
    screen.blit(t,(rect[0]+(rect[2]-t.get_width())//2,rect[1]+(rect[3]-t.get_height())//2))

def main():
    global running
    pygame.init()
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Livox Mid-360  —  Live Map")
    font_b = pygame.font.SysFont('monospace',20,bold=True)
    font   = pygame.font.SysFont('monospace',15)
    font_s = pygame.font.SysFont('monospace',12)
    clock  = pygame.time.Clock()

    threading.Thread(target=udp_recv,daemon=True).start()

    mode  = "2D"
    yaw   = 0.0
    pitch = 30.0
    zoom  = 0.045
    drag  = False
    lmpos = (0,0)

    BTN_W,BTN_H = 110,34
    btn_2d = pygame.Rect(W//2-BTN_W-8, H-PANEL_H+9, BTN_W, BTN_H)
    btn_3d = pygame.Rect(W//2+8,       H-PANEL_H+9, BTN_W, BTN_H)

    while running:
        clock.tick(30)
        now = time.monotonic()

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            elif ev.type==pygame.KEYDOWN:
                if   ev.key==pygame.K_q:  running=False
                elif ev.key==pygame.K_F1: mode="2D"
                elif ev.key==pygame.K_F2: mode="3D"
            elif ev.type==pygame.MOUSEBUTTONDOWN:
                if   btn_2d.collidepoint(ev.pos): mode="2D"
                elif btn_3d.collidepoint(ev.pos): mode="3D"
                elif ev.button==1: drag=True; lmpos=ev.pos
                elif ev.button==4 and mode=="3D": zoom=min(zoom*1.1,2.0)
                elif ev.button==5 and mode=="3D": zoom=max(zoom/1.1,0.005)
            elif ev.type==pygame.MOUSEBUTTONUP:
                if ev.button==1: drag=False
            elif ev.type==pygame.MOUSEMOTION:
                if drag and mode=="3D":
                    dx=ev.pos[0]-lmpos[0]
                    dy=ev.pos[1]-lmpos[1]
                    yaw   += dx*0.12          # slow left/right
                    pitch  = max(-89,min(89,pitch+dy*0.12))  # slow up/down
                    lmpos  = ev.pos

        # Expire old scans (keep 0.5s for 2D, 0.1s for 3D)
        expire = 0.1 if mode=="2D" else 0.1
        with scan_lk:
            while scans and now-scans[0][0]>expire: scans.popleft()
            snap = list(scans)
        with scan_lk: pk=pkt_cnt

        # ════════ 2D ═════════════════════════════════════
        if mode=="2D":
            screen.fill((255,255,255))

            # Grid rings
            for r_m,lbl in [(0.5,"0.5m"),(1,"1m"),(1.5,"1.5m"),(2,"2m")]:
                r_px=int(r_m*1000*SCALE2D)
                pygame.draw.circle(screen,(170,170,170) if r_m<2 else (90,90,90),(CX2,CY2),r_px,1)
                screen.blit(font_s.render(lbl,True,(120,120,120)),(CX2+r_px+3,CY2-9))
            pygame.draw.line(screen,(220,220,220),(0,CY2),(W,CY2),1)
            pygame.draw.line(screen,(220,220,220),(CX2,0),(CX2,VH),1)

            # Draw only points from the last `expire` window, directly at their
            # current sensor-relative position — nothing is cached across
            # frames, so moving/rotating the lidar can't leave stale dots
            # behind (no more ghost trails when you pick it up and move it).
            total_drawn=0
            for ts,xmm,ymm,zmm in snap:
                dist=np.sqrt(xmm**2+ymm**2)
                px=(CX2+xmm*SCALE2D).astype(np.int32)
                py=(CY2-ymm*SCALE2D).astype(np.int32)
                ok=(px>=0)&(px<W)&(py>=0)&(py<VH)&(dist<=2000)
                px=px[ok]; py=py[ok]; df=dist[ok]
                t=np.clip(1.0-df/2000.0,0,1)
                r=(t*255).astype(np.int32); g=(t*(1-t)*320).astype(np.int32); b=((1-t)*220).astype(np.int32)
                for i in range(len(px)):
                    pygame.draw.circle(screen,(int(r[i]),int(g[i]),int(b[i])),(px[i],py[i]),2)
                total_drawn+=len(px)

            # Sensor dot + arrow
            pygame.draw.circle(screen,(200,0,0),(CX2,CY2),9)
            pygame.draw.line(screen,(0,160,0),(CX2,CY2),(CX2,CY2-34),3)
            pygame.draw.polygon(screen,(0,160,0),[(CX2,CY2-46),(CX2-6,CY2-34),(CX2+6,CY2-34)])
            screen.blit(font_s.render("FWD",True,(0,130,0)),(CX2+11,CY2-44))

            screen.blit(font_b.render("TOP-DOWN  2m MAP",True,(0,0,110)),(12,8))
            screen.blit(font.render(f"Pkts:{pk:,}  Pts:{total_drawn:,}  FPS:{clock.get_fps():.0f}",True,(0,0,160)),(12,34))

        # ════════ 3D ═════════════════════════════════════
        else:
            screen.fill((8,10,20))

            # Simple axis cross at origin for orientation
            yr=math.radians(yaw); pr=math.radians(pitch)
            def p1(x,y,z):
                sx,sy,_=project3d(np.array([x],np.float32),
                                   np.array([y],np.float32),
                                   np.array([z],np.float32),
                                   yr,pr,zoom,W//2,VH//2)
                return int(sx[0]),int(sy[0])
            o=p1(0,0,0)
            for (ax,ay,az),col,lbl in [((2000,0,0),(255,80,80),"X"),
                                        ((0,2000,0),(80,255,80),"Y"),
                                        ((0,0,2000),(80,80,255),"Z")]:
                ep=p1(ax,ay,az)
                pygame.draw.line(screen,col,o,ep,2)
                screen.blit(font_s.render(lbl,True,col),(ep[0]+4,ep[1]-8))
            pygame.draw.circle(screen,(255,60,60),o,8)
            screen.blit(font_s.render("YOU",True,(255,100,100)),(o[0]+10,o[1]-8))

            # Points
            all_x=[]; all_y=[]; all_z=[]
            for ts,xmm,ymm,zmm in snap:
                all_x.append(xmm); all_y.append(ymm); all_z.append(zmm)

            total=0
            if all_x:
                xmm=np.concatenate(all_x)
                ymm=np.concatenate(all_y)
                zmm=np.concatenate(all_z)

                sx,sy,depth=project3d(xmm,ymm,zmm,yr,pr,zoom,W//2,VH//2)
                ok=(sx>=0)&(sx<W)&(sy>=0)&(sy<VH)
                sx=sx[ok]; sy=sy[ok]; depth=depth[ok]
                dm=np.sqrt(xmm[ok]**2+ymm[ok]**2+zmm[ok]**2)
                total=len(sx)

                # Sort back-to-front
                order=np.argsort(-depth)
                sx=sx[order]; sy=sy[order]; dm=dm[order]

                # Red=close Green=far
                t=np.clip(dm/6000.0,0,1)
                rc=np.clip(((1-t)*255).astype(int),0,255)
                gc=np.clip((t*255).astype(int),0,255)
                bc=np.zeros(len(t),dtype=int)

                for i in range(len(sx)):
                    pygame.draw.circle(screen,(int(rc[i]),int(gc[i]),0),(sx[i],sy[i]),2)

            screen.blit(font_b.render("3D POINT CLOUD",True,(0,220,255)),(12,8))
            screen.blit(font.render(f"Pkts:{pk:,}  Pts:{total:,}  FPS:{clock.get_fps():.0f}",True,(0,180,230)),(12,34))
            screen.blit(font_s.render(f"Yaw:{yaw%360:.0f}°  Pitch:{pitch:.0f}°  Drag=rotate  Scroll=zoom  RED=close  GREEN=far",
                                       True,(0,100,160)),(12,56))

        # ── Panel ─────────────────────────────────────────
        pygame.draw.rect(screen,(232,232,232),(0,VH,W,PANEL_H))
        pygame.draw.line(screen,(180,180,180),(0,VH),(W,VH),1)
        draw_btn(screen,font,"  2D  ",btn_2d,mode=="2D")
        draw_btn(screen,font,"  3D  ",btn_3d,mode=="3D")
        screen.blit(font_s.render("F1=2D  F2=3D  Q=quit",True,(130,130,130)),(12,VH+18))

        pygame.display.flip()

    running=False
    pygame.quit()

if __name__=="__main__":
    main()
