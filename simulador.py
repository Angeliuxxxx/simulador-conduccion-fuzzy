"""
Simulador RetroNeon AI Driving (Mamdani fuzzy)
Guarda como retro_neon_fuzzy_sim.py
Requisitos: pygame, numpy, scikit-fuzzy, pandas (opcional)
"""

import os
import sys
import math
import random
import time
from datetime import datetime
import csv

import numpy as np
import pandas as pd

import pygame
from pygame import gfxdraw

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------- Config --------------------
SCREEN_W = 1200
SCREEN_H = 600
FPS = 60

ASSETS_DIR = "assets"  # carpeta opcional para sprites

# -------------------- Fuzzy controller (Mamdani) --------------------
class FuzzyController:
    def __init__(self):
        # Universos
        self.vel_univ = np.arange(0, 121, 1)     # km/h
        self.dist_univ = np.arange(0, 101, 1)    # m
        self.vis_univ = np.arange(0, 101, 1)     # %
        self.act_univ = np.arange(0, 101, 1)     # 0..100

        # Variables difusas
        self.vel = ctrl.Antecedent(self.vel_univ, 'velocidad')
        self.dist = ctrl.Antecedent(self.dist_univ, 'distancia')
        self.vis = ctrl.Antecedent(self.vis_univ, 'visibilidad')
        self.act = ctrl.Consequent(self.act_univ, 'accion')

        # Membresías
        self.vel['baja'] = fuzz.trimf(self.vel_univ, [0, 0, 50])
        self.vel['media'] = fuzz.trimf(self.vel_univ, [30, 60, 90])
        self.vel['alta'] = fuzz.trimf(self.vel_univ, [70, 120, 120])

        self.dist['corta'] = fuzz.trimf(self.dist_univ, [0, 0, 30])
        self.dist['media'] = fuzz.trimf(self.dist_univ, [20, 50, 80])
        self.dist['larga'] = fuzz.trimf(self.dist_univ, [60, 100, 100])

        self.vis['baja'] = fuzz.trimf(self.vis_univ, [0, 0, 40])
        self.vis['media'] = fuzz.trimf(self.vis_univ, [30, 60, 90])
        self.vis['alta'] = fuzz.trimf(self.vis_univ, [70, 100, 100])

        self.act['frenar'] = fuzz.trimf(self.act_univ, [0, 0, 40])
        self.act['mantener'] = fuzz.trimf(self.act_univ, [30, 50, 70])
        self.act['acelerar'] = fuzz.trimf(self.act_univ, [60, 100, 100])

        # Reglas
        rules = [
            ctrl.Rule(self.dist['corta'] & self.vel['alta'], self.act['frenar']),
            ctrl.Rule(self.dist['corta'] & self.vel['media'], self.act['frenar']),
            ctrl.Rule(self.dist['corta'] & self.vel['baja'], self.act['mantener']),
            ctrl.Rule(self.dist['media'] & self.vel['alta'], self.act['frenar']),
            ctrl.Rule(self.dist['media'] & self.vel['media'], self.act['mantener']),
            ctrl.Rule(self.dist['media'] & self.vel['baja'], self.act['mantener']),
            ctrl.Rule(self.dist['larga'] & self.vis['alta'] & self.vel['baja'], self.act['acelerar']),
            ctrl.Rule(self.dist['larga'] & self.vis['alta'] & self.vel['media'], self.act['acelerar']),
            ctrl.Rule(self.dist['larga'] & self.vis['media'], self.act['mantener']),
            ctrl.Rule(self.vis['baja'], self.act['frenar']),
            ctrl.Rule(self.vel['baja'] & self.dist['corta'], self.act['mantener']),
            ctrl.Rule(self.vel['alta'] & self.vis['media'] & self.dist['media'], self.act['frenar']),
        ]

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def compute(self, v, d, vis):
        # clamp values
        v = float(max(0, min(120, v)))
        d = float(max(0, min(100, d)))
        vis = float(max(0, min(100, vis)))
        self.sim.input['velocidad'] = v
        self.sim.input['distancia'] = d
        self.sim.input['visibilidad'] = vis
        try:
            self.sim.compute()
            return float(self.sim.output['accion'])
        except Exception as e:
            print("Fuzzy error:", e)
            return 50.0

# -------------------- UI helper: Slider --------------------
class Slider:
    def __init__(self, rect, minv, maxv, val, label, color=(200,200,200)):
        self.rect = pygame.Rect(rect)
        self.minv = minv
        self.maxv = maxv
        self.value = val
        self.label = label
        self.dragging = False
        self.color = color
        self.handle_radius = 10

    def draw(self, surf, font):
        # background line
        x,y,w,h = self.rect
        line_y = y + h//2
        pygame.draw.rect(surf, (50,50,50), (x, line_y-3, w, 6), border_radius=3)
        # handle position
        rel = (self.value - self.minv) / (self.maxv - self.minv)
        hx = x + int(rel * w)
        # neon effect
        for i,alpha in enumerate([40,80,140]):
            col = (100, 60+i*40, 200)
            s = pygame.Surface((self.handle_radius*3, self.handle_radius*3), pygame.SRCALPHA)
            pygame.draw.circle(s, col + (40 - i*10,), (self.handle_radius*1, self.handle_radius*1), self.handle_radius + (2-i))
            surf.blit(s, (hx - self.handle_radius -1, line_y - self.handle_radius -1))
        # main handle
        pygame.draw.circle(surf, (255,255,255), (hx, line_y), self.handle_radius)
        # label & value
        lbl = font.render(f"{self.label}: {self.value:.0f}", True, (230,230,230))
        surf.blit(lbl, (x, y - 22))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mx,my = event.pos
                x,y,w,h = self.rect
                line_y = y + h//2
                rel = (self.value - self.minv) / (self.maxv - self.minv)
                hx = x + int(rel * w)
                if (mx-hx)**2 + (my-line_y)**2 <= (self.handle_radius+5)**2:
                    self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx,my = event.pos
            x,y,w,h = self.rect
            clamped = max(x, min(x+w, mx))
            rel = (clamped - x) / w
            self.value = self.minv + rel * (self.maxv - self.minv)

# -------------------- Simulation --------------------
class RetroNeonSim:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Automax")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.bigfont = pygame.font.SysFont("Consolas", 28, bold=True)

        # load assets
        self.car_sprite = None
        self.obst_sprite = None
        self.bg_sprite = None
        self.try_load_assets()

        # fuzzy controller
        self.fuzzy = FuzzyController()

        # simulation state
        self.speed = 40.0        # km/h (actual)
        self.target_speed = 40.0 # slider initial
        self.px_per_m = 5.0      # visual scale: pixels per meter
        self.car_x = SCREEN_W // 2 - 64
        self.car_y = SCREEN_H - 180
        self.car_w = 128
        self.car_h = 64
        self.obst_distance_m = 40.0  # meters (slider)
        self.visibility = 100.0       # %
        self.obst_base_x = self.car_x + 400  # base position for obstacle drawing
        self.time = 0.0

        # velocities and physics
        self.dt = 1.0 / FPS
        self.accel = 0.0

        # UI sliders
        sx = 40; sw = 380; sy = 60
        self.slider_speed = Slider((sx, sy, sw, 30), 0, 120, self.speed, "Velocidad (km/h)")
        self.slider_dist = Slider((sx, sy+80, sw, 30), 0, 100, self.obst_distance_m, "Distancia (m)")
        self.slider_vis = Slider((sx, sy+160, sw, 30), 0, 100, self.visibility, "Visibilidad (%)")

        # toggle mode
        self.demo_mode = False
        self.demo_timer = 0.0

        # particles list
        self.particles = []

        # brake light
        self.brake_on = False

        # log
        self.log = []

        # for neon road animation: lines positions
        self.road_offset = 0.0

        # run
        self.running = True

    def try_load_assets(self):
        try:
            car_path = os.path.join(ASSETS_DIR, "car.png")
            if os.path.exists(car_path):
                self.car_sprite = pygame.image.load(car_path).convert_alpha()
                self.car_sprite = pygame.transform.smoothscale(self.car_sprite, (self.car_w, self.car_h))
        except Exception as e:
            print("Car sprite load error:", e)
            self.car_sprite = None
        try:
            obst_path = os.path.join(ASSETS_DIR, "obstacle.png")
            if os.path.exists(obst_path):
                self.obst_sprite = pygame.image.load(obst_path).convert_alpha()
        except Exception as e:
            print("Obstacle sprite load error:", e)
            self.obst_sprite = None
        try:
            bg_path = os.path.join(ASSETS_DIR, "background.png")
            if os.path.exists(bg_path):
                self.bg_sprite = pygame.image.load(bg_path).convert()
                self.bg_sprite = pygame.transform.smoothscale(self.bg_sprite, (SCREEN_W, SCREEN_H))
        except Exception as e:
            print("BG sprite load error:", e)
            self.bg_sprite = None

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.dt = dt
            self.time += dt
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            self.slider_speed.handle_event(e)
            self.slider_dist.handle_event(e)
            self.slider_vis.handle_event(e)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                if e.key == pygame.K_SPACE:
                    # toggle demo
                    self.demo_mode = not self.demo_mode
                    print("Demo mode:", self.demo_mode)
                if e.key == pygame.K_s:
                    # save log csv
                    self.save_log_csv()
                    print("Log saved.")
                if e.key == pygame.K_r:
                    # reset
                    self.reset_sim()
            # mouse interactivity for dragging obstacle (optional)
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    mx,my = e.pos
                    # if clicking obstacle area, allow moving
                    if self.obstacle_screen_rect().collidepoint(mx,my):
                        self.dragging_obstacle = True
            if e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self.dragging_obstacle = False
            if e.type == pygame.MOUSEMOTION:
                if getattr(self, "dragging_obstacle", False):
                    mx,my = e.pos
                    # compute new distance based on mouse x
                    car_front_x = self.car_x + self.car_w
                    dx = mx - car_front_x
                    new_dist_m = max(0.0, dx / self.px_per_m)
                    self.obst_distance_m = min(100.0, new_dist_m)
                    self.slider_dist.value = self.obst_distance_m

    def reset_sim(self):
        self.speed = 40.0
        self.slider_speed.value = self.speed
        self.obst_distance_m = 40.0
        self.slider_dist.value = self.obst_distance_m
        self.visibility = 100.0
        self.slider_vis.value = self.visibility
        self.particles = []
        self.brake_on = False
        self.log = []
        self.time = 0.0

    def update(self, dt):
        # if user moves sliders, read them (manual input)
        self.speed = self.slider_speed.value  # interpret slider as current speed reading if user sets it
        self.obst_distance_m = self.slider_dist.value
        self.visibility = self.slider_vis.value

        # demo mode: override sliders gradually
        if self.demo_mode:
            # change targets smoothly
            self.demo_timer += dt
            if self.demo_timer > 2.0:
                self.demo_timer = 0.0
                # random scenario
                self.slider_speed.value = random.uniform(20, 100)
                self.slider_dist.value = random.uniform(5, 90)
                self.slider_vis.value = random.uniform(20, 100)
                # assign to current state (simulate sensor)
                self.speed = self.slider_speed.value
                self.obst_distance_m = self.slider_dist.value
                self.visibility = self.slider_vis.value

        # --- Fuzzy decision (compute action) ---
        action_val = self.fuzzy.compute(self.speed, self.obst_distance_m, self.visibility)  # 0-100

        # Map fuzzy action to acceleration (crisp)
        if action_val < 35:
            accel = -60.0     # strong brake (km/h per s approx)
            self.brake_on = True
            self.add_particles(intensity=1.2, direction=-1)
            action_text = "FRENAR"
        elif action_val > 65:
            accel = 18.0      # accelerate
            self.brake_on = False
            self.add_particles(intensity=1.6, direction=1)
            action_text = "ACELERAR"
        else:
            accel = -6.0      # slight decel / maintain
            self.brake_on = False
            action_text = "MANTENER"

        # integrate speed change (approx)
        # Note: in this simulation we let sliders represent target/current reading,
        # but we'll show visual change in animation by smoothing speed toward a physics-mode speed
        # To visualize the fuzzy decision affecting speed, we will modify displayed_speed progressively.
        # We'll store displayed_speed separately.
        if not hasattr(self, 'display_speed'):
            self.display_speed = self.speed
        # accelerate display speed by accel
        self.display_speed += accel * dt
        # apply some damping toward slider's base if user set slider
        # let displayed speed relax to slider value slowly (so user input is immediate but decision shows effect)
        relax = 1.8  # relaxation factor
        self.display_speed += (self.speed - self.display_speed) * min(1.0, relax * dt)

        # clamp
        self.display_speed = max(0.0, min(120.0, self.display_speed))

        # move road offset by speed (simulate forward motion)
        # convert km/h to m/s -> px movement
        vel_m_s = (self.display_speed * 1000.0) / 3600.0
        px_move = vel_m_s * self.px_per_m * dt
        self.road_offset = (self.road_offset + px_move) % 60  # spacing of stripes

        # update particles physics
        self.update_particles(dt)

        # update logging
        self.log.append({
            "time": round(self.time,3),
            "slider_speed": round(self.speed,3),
            "display_speed": round(self.display_speed,3),
            "distance_m": round(self.obst_distance_m,3),
            "visibility": round(self.visibility,3),
            "action_val": round(action_val,3),
            "action_text": action_text
        })

    def add_particles(self, intensity=1.0, direction=1):
        # add a few particles near rear (direction -1 for braking/backwards)
        for i in range(int(1 + intensity*2)):
            x = self.car_x + (self.car_w/2) + random.uniform(-10,10)
            y = self.car_y + self.car_h + random.uniform(0,8)
            life = random.uniform(0.4, 1.0)
            vx = random.uniform(10,60) * (0.02 * (-direction))
            vy = random.uniform(-10,5) * 0.02
            self.particles.append([x,y,life,vx,vy])

    def update_particles(self, dt):
        newp = []
        for p in self.particles:
            x,y,life,vx,vy = p
            life -= dt
            x += vx * dt * 100
            y += vy * dt * 100
            if life > 0:
                newp.append([x,y,life,vx,vy])
        self.particles = newp

    def obstacle_screen_rect(self):
        # compute obstacle size based on distance: closer => bigger
        d = max(0.1, self.obst_distance_m)
        scale = 1.0 + max(0, (40.0 - d)/60.0) * 2.0  # heuristic scale
        obst_w = int(50 * scale)
        obst_h = int(50 * scale)
        car_front_x = self.car_x + self.car_w
        # obstacle draw x (distance -> px)
        off_px = int(self.obst_distance_m * self.px_per_m)
        x = car_front_x + off_px
        y = self.car_y - int(obst_h * 0.6)
        return pygame.Rect(x, y, obst_w, obst_h)

    def draw_neon_text(self, surf, text, pos, size=24, glow_color=(120,60,220)):
        # draw outer glow by drawing text multiple times
        f = pygame.font.SysFont("Consolas", size, bold=True)
        base = f.render(text, True, (255,255,255))
        x,y = pos
        # glow
        for i,alpha in enumerate([60,30,15]):
            s = f.render(text, True, (glow_color[0], glow_color[1], glow_color[2]))
            s.set_alpha(80 - i*20)
            surf.blit(s, (x- i -2, y - i -2))
        surf.blit(base, pos)

    def draw(self):
        s = self.screen
        # background
        if self.bg_sprite:
            s.blit(self.bg_sprite, (0,0))
        else:
            # gradient sky/neon
            s.fill((10,6,20))
            for i in range(120):
                c = 12 + i//6
                pygame.draw.rect(s, (c*2, 0, c*3), (0, i*5, SCREEN_W, 5))
        # road area
        road_h = 220
        ry = SCREEN_H - road_h
        pygame.draw.rect(s, (18,18,18), (0, ry, SCREEN_W, road_h))

        # lane marks (neon dashed)
        dash_w = 40
        gap = 20
        lane_y = ry + road_h//2
        color_line = (255,200,60)
        offset = int(self.road_offset)
        for x in range(-dash_w*2, SCREEN_W + dash_w*2, dash_w + gap):
            xx = x - offset
            pygame.draw.rect(s, color_line, (xx, lane_y-4, dash_w, 8))

        # draw obstacle (scaled)
        rect = self.obstacle_screen_rect()
        if self.obst_sprite:
            sprite = pygame.transform.smoothscale(self.obst_sprite, (rect.w, rect.h))
            s.blit(sprite, (rect.x, rect.y))
        else:
            # neon rectangle
            pygame.gfxdraw.box(s, rect, (215,75,75))
            pygame.gfxdraw.rectangle(s, rect, (255,120,120))

        # draw car (center-bottom)
        if self.car_sprite:
            s.blit(self.car_sprite, (self.car_x, self.car_y))
        else:
            # car body
            pygame.gfxdraw.filled_rounded_rect = None
            pygame.draw.rect(s, (40,110,220), (self.car_x, self.car_y, self.car_w, self.car_h), border_radius=12)
            pygame.draw.rect(s, (18,18,18), (self.car_x+8, self.car_y+8, self.car_w-16, self.car_h-20), border_radius=8)
            # wheels
            pygame.draw.ellipse(s, (20,20,20), (self.car_x+10, self.car_y+self.car_h-12, 28,14))
            pygame.draw.ellipse(s, (20,20,20), (self.car_x+self.car_w-38, self.car_y+self.car_h-12, 28,14))

        # brake lights
        if self.brake_on:
            bx = self.car_x + self.car_w - 12
            by = self.car_y + 16
            pygame.draw.rect(s, (255,10,10), (bx, by, 8, 10))
            pygame.draw.rect(s, (255,10,10), (self.car_x+4, by, 8, 10))

        # particles
        for x,y,life,vx,vy in self.particles:
            alpha = int(255 * (life/1.0))
            col = (255,180,60, alpha)
            surf = pygame.Surface((6,6), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255,180,60, alpha), (3,3), 3)
            s.blit(surf, (x,y))

        # HUD panel (semi-transparent)
        hud = pygame.Surface((380, 320), pygame.SRCALPHA)
        hud.fill((10,10,10,160))
        s.blit(hud, (20, 30))

        # Draw sliders
        self.slider_speed.draw(s, self.font)
        self.slider_dist.draw(s, self.font)
        self.slider_vis.draw(s, self.font)

        # Draw labels and neon texts
        # top-right status
        self.draw_neon_text(s, "AutoMax", (450, 18), size=32, glow_color=(170,60,220))
        # info block (right)
        info_x = 820
        info_y = 60
        # Real values
        dsurf = self.font.render(f"Velocidad (slider): {self.speed:.1f} km/h (press S to save, SPACE demo, R reset)", True, (220,220,220))
        s.blit(dsurf, (info_x, info_y))
        d2 = self.font.render(f"Distancia: {self.obst_distance_m:.1f} m   Visibilidad: {self.visibility:.0f}%", True, (220,220,220))
        s.blit(d2, (info_x, info_y + 28))
        # last action from log
        if self.log:
            last = self.log[-1]
            act = last["action_text"]
            action_val = last["action_val"]
            txt = f"Última acción: {act}  ({action_val})"
            actsurf = self.bigfont.render(txt, True, (255,200,120))
            s.blit(actsurf, (info_x, info_y + 70))

        # draw small instruction
        inst = self.font.render("Ajusta sliders o presiona SPACE para Demo automático", True, (180,180,180))
        s.blit(inst, (20, 370))

        # draw a legend for controls
        legend = self.font.render("Teclas: SPACE Demo | S guardar CSV | R reset | ESC salir", True, (180,180,180))
        s.blit(legend, (20, SCREEN_H - 30))

        pygame.display.flip()

    def save_log_csv(self):
        if not self.log:
            return
        os.makedirs("results", exist_ok=True)
        fname = f"results/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.log)
        df.to_csv(fname, index=False)
        print("Saved", fname)

# -------------------- Main --------------------
def main():
    sim = RetroNeonSim()
    sim.run()

if __name__ == "__main__":
    main()
