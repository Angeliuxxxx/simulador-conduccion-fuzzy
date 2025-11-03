
"""
Simulador AUTOMAX (Mamdani fuzzy)
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

        # --- MEJORA DE REGLAS DE IA ---
        rules = [
            ctrl.Rule(self.vis['baja'], self.act['frenar']),
            ctrl.Rule(self.dist['corta'] & (self.vel['media'] | self.vel['alta']), self.act['frenar']),
            ctrl.Rule(self.dist['corta'] & self.vel['baja'], self.act['mantener']),
            ctrl.Rule(self.dist['media'] & self.vel['alta'], self.act['frenar']),
            ctrl.Rule(self.dist['media'] & self.vel['media'], self.act['mantener']),
            ctrl.Rule(self.dist['media'] & self.vel['baja'], self.act['mantener']),
            ctrl.Rule(self.dist['larga'] & self.vis['alta'] & (self.vel['baja'] | self.vel['media']), self.act['acelerar']),
            ctrl.Rule(self.dist['larga'] & self.vis['alta'] & self.vel['alta'], self.act['mantener']),
            ctrl.Rule(self.dist['larga'] & self.vis['media'], self.act['mantener']),
        ]
        # --- FIN DE MEJORA DE REGLAS ---

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def compute(self, v, d, vis):
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
        x,y,w,h = self.rect
        line_y = y + h//2
        pygame.draw.rect(surf, (50,50,50), (x, line_y-3, w, 6), border_radius=3)
        rel = (self.value - self.minv) / (self.maxv - self.minv)
        hx = x + int(rel * w)
        for i,alpha in enumerate([40,80,140]):
            col = (100, 60+i*40, 200)
            s = pygame.Surface((self.handle_radius*3, self.handle_radius*3), pygame.SRCALPHA)
            pygame.draw.circle(s, col + (40 - i*10,), (self.handle_radius*1, self.handle_radius*1), self.handle_radius + (2-i))
            surf.blit(s, (hx - self.handle_radius -1, line_y - self.handle_radius -1))
        pygame.draw.circle(surf, (255,255,255), (hx, line_y), self.handle_radius)
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
        

        # fuzzy controller
        self.fuzzy = FuzzyController()

        # simulation state
        self.speed = 40.0        # km/h (actual)
        self.target_speed = 40.0 # slider initial
        self.px_per_m = 5.0      # visual scale: pixels per meter
        
        # *** CAMBIO: Dimensiones del coche unificadas ***
        # car visual (dimensiones de la vista trasera)
        self.car_vis_w = 180
        self.car_vis_h = 140
        self.car_x = SCREEN_W // 2 - self.car_vis_w // 2
        self.car_y = SCREEN_H - self.car_vis_h - 18
        self.car_w = self.car_vis_w # Usamos el ancho visual
        self.car_h = self.car_vis_h # Usamos el alto visual
        

        self.try_load_assets()

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
                # *** CAMBIO: Usar dimensiones de vista trasera si se carga car.png ***
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
            # --- Cerrar ventana ---
            if e.type == pygame.QUIT:
                    self.running = False

            # --- Sliders ---
            self.slider_speed.handle_event(e)
            self.slider_dist.handle_event(e)
            self.slider_vis.handle_event(e)

            # --- Teclado ---
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_SPACE:
                    # Cambia el modo demo (automático)
                    self.demo_mode = not self.demo_mode
                    print("Demo mode:", self.demo_mode)
                elif e.key == pygame.K_s:
                    self.save_log_csv()
                    print("Log guardado.")
                elif e.key == pygame.K_r:
                    self.reset_sim()

            # --- Mouse: arrastrar obstáculo manualmente ---
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:  # botón izquierdo
                    mx, my = e.pos
                    if self.obstacle_screen_rect().collidepoint(mx, my):
                        self.dragging_obstacle = True

            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self.dragging_obstacle = False

            elif e.type == pygame.MOUSEMOTION and getattr(self, "dragging_obstacle", False):
                # Solo mover obstáculo con el mouse
                mx, my = e.pos
                SCREEN_W, SCREEN_H = self.screen.get_size()
                road_horizon_y = 120
                car_front_y = self.car_y
                my_clamped = max(road_horizon_y + 10, min(car_front_y - 20, my))

                # Mapea la posición del mouse en pantalla a distancia (0..100 m)
                t = (my_clamped - road_horizon_y) / max(1.0, (car_front_y - road_horizon_y - 40))
                new_dist_m = max(0.0, min(100.0, (1.0 - t) * 100.0))
                self.obst_distance_m = new_dist_m
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

    # *** CAMBIO: Función 'update' simplificada (lógica arreglada) ***
    def update(self, dt):
        # if user moves sliders, read them (manual input)
        self.speed = self.slider_speed.value
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
            # *** LLAMADA A PARTICULAS (AHORA USA self.car_w) ***
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
        if not hasattr(self, 'display_speed'):
            self.display_speed = self.speed
        self.display_speed += accel * dt
        relax = 1.8
        self.display_speed += (self.speed - self.display_speed) * min(1.0, relax * dt)
        self.display_speed = max(0.0, min(120.0, self.display_speed))

        # move road offset by speed (simulate forward motion)
        vel_m_s = (self.display_speed * 1000.0) / 3600.0
        px_move = vel_m_s * self.px_per_m * dt
        self.road_offset = (self.road_offset + px_move) % 60
        
        # --- LÍNEAS DE FÍSICA EN CONFLICTO (BORRADAS) ---
        # El obstáculo ya no se mueve por su cuenta.
        # El slider de distancia es la única fuente de verdad.
        
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

    # *** CAMBIO: Función 'add_particles' (posición arreglada) ***
    def add_particles(self, intensity=1.0, direction=1):
        # Genera partículas en las posiciones de las llantas traseras
        
        # Posiciones X aproximadas de las llantas (basadas en las luces de freno)
        llanta_izq_x = self.car_x + 30 + random.uniform(-2, 2)
        llanta_der_x = self.car_x + self.car_w - 50 + random.uniform(-2, 2)
        
        # Posición Y (justo en la parte inferior del coche)
        y = self.car_y + self.car_h - 25 + random.uniform(0, 8)

        # Genera la mitad de partículas en cada llanta
        for i in range(int(1 + intensity)): # Mitad en la izquierda
            life = random.uniform(0.4, 1.0)
            vx = random.uniform(10,60) * (0.02 * (-direction))
            vy = random.uniform(-10,5) * 0.02
            self.particles.append([llanta_izq_x, y, life, vx, vy])
            
        for i in range(int(1 + intensity)): # Mitad en la derecha
            life = random.uniform(0.4, 1.0)
            vx = random.uniform(10,60) * (0.02 * (-direction))
            vy = random.uniform(-10,5) * 0.02
            self.particles.append([llanta_der_x, y, life, vx, vy])

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
        """
        Map obst_distance_m (0..100) to a rectangle on the perspective road.
        - 100m -> near horizon (small)
        - 0m   -> very close to the car (large)
        """
        SCREEN_W, SCREEN_H = self.screen.get_size()
        center_x = SCREEN_W // 2

        # horizon and car front positions (fixed)
        road_horizon_y = 120
        car_front_y = self.car_y  # car's top Y is used as reference for "close" region

        # normalized distance t in [0,1] where 1.0 means far (horizon), 0.0 is at car front
        d = max(0.0, min(100.0, self.obst_distance_m))
        t = d / 100.0

        # compute y: far (t=1) -> near horizon; close (t=0) -> near car_front_y - small offset
        max_travel = max(20.0, (car_front_y - road_horizon_y - 40.0))
        y = int(road_horizon_y + (1.0 - t) * max_travel)

        # size scales inversely with distance (close -> bigger)
# size scales inversely with distance (close -> bigger)
        width = int(max(24, 24 + (1.0 - t) * 120))   # ancho más grande
        height = int(max(20, 20 + (1.0 - t) * 80))   # alto más moderado

        x = center_x - width // 2
        return pygame.Rect(x, y, width, height)


    def draw_neon_text(self, surf, text, pos, size=24, glow_color=(120,60,220)):
        f = pygame.font.SysFont("Consolas", size, bold=True)
        base = f.render(text, True, (255,255,255))
        x,y = pos
        for i,alpha in enumerate([60,30,15]):
            s = f.render(text, True, (glow_color[0], glow_color[1], glow_color[2]))
            s.set_alpha(80 - i*20)
            surf.blit(s, (x- i -2, y - i -2))
        surf.blit(base, pos)

    def draw(self):
        s = self.screen
        SCREEN_W, SCREEN_H = s.get_size()
        center_x = SCREEN_W // 2

        # ---------- Fondo dinámico (más claro a mayor velocidad) ----------
        # speed_factor in [0..1]
      # ---------- Fondo dinámico (gris azulado más notorio con la velocidad) ----------
# ---------- Fondo dinámico (gris azulado que se oscurece notablemente con la velocidad) ----------
        speed_factor = min(1.0, getattr(self, "display_speed", 0.0) / 120.0)

        # Colores base (más claros a baja velocidad)
        # A altas velocidades se vuelven mucho más oscuros
        top_color = int( 100 - speed_factor * 150)  # antes 160 - 100
        bot_color = int(255 - speed_factor * 140)  # antes 190 - 90

        for i in range(SCREEN_H):
            t = i / SCREEN_H  # proporción vertical
            
            # Combinación con un tono más azulado y más contraste
            r = int((top_color * (1 - t) + (bot_color - 30) * t))
            g = int((top_color * 0.9 * (1 - t) + (bot_color - 40) * t))
            b = int((top_color * 1.4 * (1 - t) + (bot_color + 50) * t))

            # Limitar valores válidos
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            pygame.draw.line(s, (r, g, b), (0, i), (SCREEN_W, i))





        # ---------- Carretera en perspectiva ----------
        road_bottom_y = SCREEN_H
        road_horizon_y = 110
        lane_color = (70, 170, 255)
        asphalt_color = (20, 20, 24)
        road_left_w = 180
        road_right_w = 180
        road_horizon_inner = 40  # half-width at horizon

        # Road polygon (asphalt)
        pygame.draw.polygon(s, asphalt_color, [
            (center_x - road_left_w, road_bottom_y),
            (center_x - road_horizon_inner, road_horizon_y),
            (center_x + road_horizon_inner, road_horizon_y),
            (center_x + road_right_w, road_bottom_y)
        ])

        # Outer lane edges (glow)
        pygame.draw.line(s, lane_color, (center_x - road_left_w, road_bottom_y), (center_x - road_horizon_inner, road_horizon_y), 3)
        pygame.draw.line(s, lane_color, (center_x + road_left_w, road_bottom_y), (center_x + road_horizon_inner, road_horizon_y), 3)

       
        # ---------- Líneas centrales verticales (autopista con movimiento suave y delgado) ----------
        dash_h = 40       # altura base de cada línea (antes 60)
        gap = 90           # espacio entre líneas (más grande para respiro visual)
        color_line = (255, 255, 200)  # color cálido tipo autopista nocturna

        # velocidad (km/h → m/s)
        vel_m_s = (getattr(self, "display_speed", 0.0) * 1000.0) / 3600.0
        move_px = vel_m_s * self.px_per_m * self.dt * 2.2  # factor de desplazamiento

        # offset de carretera (continuo)
        self.road_offset = (self.road_offset + move_px) % (dash_h + gap)
        offset_pix = int(self.road_offset)

        # centro de la carretera
        lane_x = center_x - 2  # alineado con eje central
        total_height = SCREEN_H - road_horizon_y

        for y in range(int(road_horizon_y - (dash_h + gap)),
                       int(SCREEN_H + (dash_h + gap)),
                       dash_h + gap):
            yy = y + offset_pix

            # ignora líneas fuera de vista
            if yy < road_horizon_y - dash_h or yy > SCREEN_H + dash_h:
                continue

            # perspectiva: se reduce con la distancia
            t = (yy - road_horizon_y) / total_height
            scale = 1.0 - min(1.0, t * 0.9)
            dash_w = int(3 * (1.0 + scale * 1.3))   # ancho reducido
            dash_h_scaled = int(dash_h * (0.5 + scale * 0.7))  # largo más corto

            # aparición / desaparición gradual
            if yy < road_horizon_y + 100:
                alpha = int(255 * ((yy - road_horizon_y) / 100.0))
            elif yy > SCREEN_H - 120:
                alpha = int(255 * ((SCREEN_H - yy) / 120.0))
            else:
                alpha = 255
            alpha = max(0, min(255, alpha))

            # color dinámico (ligero tono azul a alta velocidad)
            brightness = int(180 + min(60, self.display_speed * 0.6))
            line_color = (brightness, brightness, 255)

            # dibujar con transparencia
            surf = pygame.Surface((dash_w, dash_h_scaled), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*line_color, alpha), (0, 0, dash_w, dash_h_scaled))
            s.blit(surf, (lane_x - dash_w // 2, yy - dash_h_scaled // 2))



        # ---------- Obstáculo (usa perspective mapping) ----------
        rect = self.obstacle_screen_rect()
        if self.obst_sprite:
            sprite = pygame.transform.smoothscale(self.obst_sprite, (rect.w, rect.h))
            s.blit(sprite, (rect.x, rect.y))
        else:
            # neon box with subtle glow
            glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
            pygame.draw.rect(glow, (215,75,75,60), (0,0,rect.w+16, rect.h+16), border_radius=6)
            s.blit(glow, (rect.x-8, rect.y-8))
            pygame.gfxdraw.box(s, rect, (215,75,75))
            pygame.gfxdraw.rectangle(s, rect, (255,120,120))

        # ---------- Car (rear view) ----------
        # *** CAMBIO: Usar variables de 'self' para dibujar el coche ***
        # place car centered near bottom
        # Usa las variables de 'self' que ya definimos en __init__
        cx, cy, cw, ch = self.car_x, self.car_y, self.car_w, self.car_h

        if self.car_sprite:
            car_img = pygame.transform.smoothscale(self.car_sprite, (cw, ch))
            s.blit(car_img, (cx, cy))
        else:
            # body
            body_color = (38, 58, 80)
            pygame.draw.rect(s, body_color, (cx + 8, cy + 8, cw - 16, ch - 16), border_radius=14)
            # roof
            pygame.draw.rect(s, (70,90,120), (cx + 18, cy + 16, cw - 36, ch // 3), border_radius=8)
            # rear lights (base)
            pygame.draw.rect(s, (150, 30, 30), (cx + 40, cy + ch - 40, 10, 1))
            pygame.draw.rect(s, (150, 30, 30), (cx + cw - 60, cy + ch - 40, 10, 1))
            # LED strip middle
            pygame.draw.rect(s, (110,140,190), (cx + cw//2 - 8, cy + ch - 28, 16, 4), border_radius=2)
            # side shadow
            pygame.draw.rect(s, (20, 20, 28), (cx, cy + 8, 8, ch - 16))
            pygame.draw.rect(s, (20, 20, 28), (cx + cw - 8, cy + 8, 8, ch - 16))

        # dynamic brake lights intensity (brake_on and how strong accel change is)
        if getattr(self, "brake_on", False):
            # stronger glow when braking
            glow_surf = pygame.Surface((cw+20, 24), pygame.SRCALPHA)
            #pygame.draw.rect(glow_surf, (255,40,40,140), (6, 0, 14, 6), border_radius=3)
            #pygame.draw.rect(glow_surf, (255,40,40,140), (cw - 26, 0, 14, 6), border_radius=3)
            s.blit(glow_surf, (cx - 6, cy + ch - 30))
            pygame.draw.rect(s, (255,12,12), (cx + 40, cy + ch - 74, 16, 6))
            pygame.draw.rect(s, (255,12,12), (cx + cw - 60, cy + ch - 74, 16, 6))

        # ---------- Partículas ----------
        for x,y,life,vx,vy in self.particles:
            alpha = int(255 * (life/1.0))
            surf = pygame.Surface((6,6), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255,180,60, alpha), (3,3), 3)
            s.blit(surf, (x,y))

        # ---------- Left HUD (semi-transparent) ----------
        hud = pygame.Surface((380, 320), pygame.SRCALPHA)
        hud.fill((10,10,10,160))
        s.blit(hud, (20, 30))

        # Draw sliders (inside HUD)
        self.slider_speed.draw(s, self.font)
        self.slider_dist.draw(s, self.font)
        self.slider_vis.draw(s, self.font)

        # ---------- Center HUD (big speed) ----------
        speed_text = self.bigfont.render(f"{getattr(self, 'display_speed', 0.0):.0f}", True, (255,255,255))
        unit_text = self.font.render("km/h", True, (190,190,190))
        s.blit(speed_text, (center_x - 20, 28))
        s.blit(unit_text, (center_x + 46, 56))

        # ---------- Side indicator bars (visibility / distance) ----------
        # --- BARRAS DE VISIBILIDAD Y DISTANCIA (abajo a la derecha) ---

        # Tamaños y margen
        vis_bar_h = 120
        dist_bar_h = 120
        bar_w = 16
        margin = 20

        # Posición base (cerca de la esquina inferior derecha)
        SCREEN_W, SCREEN_H = s.get_size()
        base_y = SCREEN_H - vis_bar_h - 80   # separación desde el borde inferior
        base_x = SCREEN_W - 100              # posición desde el borde derecho

        # =====================
        # VISIBILITY BAR
        # =====================
        vis_x = base_x
        vis_y = base_y
        pygame.draw.rect(s, (40, 40, 40), (vis_x, vis_y, bar_w, vis_bar_h), border_radius=4)

        # Altura proporcional según visibilidad
        vis_fill_h = int((self.visibility / 100.0) * vis_bar_h)
        grad_surf = pygame.Surface((bar_w, vis_fill_h))

        # Gradiente de color (rojo→verde)
        for yy in range(vis_fill_h):
            t = yy / max(1, vis_fill_h - 1)
            r = int(220 * (1 - t) + 60 * t)
            g = int(60 * (1 - t) + 200 * t)
            b = int(60 * (1 - t) + 80 * t)
            pygame.draw.line(grad_surf, (r, g, b), (0, vis_fill_h - 1 - yy), (bar_w, vis_fill_h - 1 - yy))

        s.blit(grad_surf, (vis_x, vis_y + (vis_bar_h - vis_fill_h)))

        # Etiqueta “Vis”
        text_vis = self.font.render("Vis", True, (200, 200, 200))
        s.blit(text_vis, (vis_x - 4, vis_y + vis_bar_h + 6))

        # =====================
        # DISTANCE BAR
        # =====================
        dist_x = base_x + 40  # separación horizontal entre barras
        dist_y = base_y
        pygame.draw.rect(s, (40, 40, 40), (dist_x, dist_y, bar_w, dist_bar_h), border_radius=4)

        # Distancia normalizada (cerca = llena, lejos = vacía)
        dnorm = max(0.0, min(1.0, 1.0 - (self.obst_distance_m / 100.0)))
        dist_fill_h = int(dnorm * dist_bar_h)
        dist_surf = pygame.Surface((bar_w, dist_fill_h))

        # Gradiente de color (amarillo→rojo)
        for yy in range(dist_fill_h):
            t = yy / max(1, dist_fill_h - 1)
            r = int(80 * (1 - t) + 220 * t)
            g = int(180 * (1 - t) + 60 * t)
            b = int(60 * (1 - t) + 70 * t)
            pygame.draw.line(dist_surf, (r, g, b), (0, dist_fill_h - 1 - yy), (bar_w, dist_fill_h - 1 - yy))

        s.blit(dist_surf, (dist_x, dist_y + (dist_bar_h - dist_fill_h)))

        # Etiqueta “Dist”
        text_dist = self.font.render("Dist", True, (200, 200, 200))
        s.blit(text_dist, (dist_x - 8, dist_y + dist_bar_h + 6))


        # ---------- Right status (last action) ----------
        if self.log:
            last = self.log[-1]
            act = last["action_text"]
            action_val = last["action_val"]
            txt = f"Acción: {act} ({action_val:.1f})"
            actsurf = self.font.render(txt, True, (255, 200, 120))
            s.blit(actsurf, (SCREEN_W - 340, 48))

        # ---------- Bottom instructions ----------
        inst = self.font.render("Presiona SPACE para modo automático | S guardar CSV | ESC salir", True, (180,180,180))
        s.blit(inst, (SCREEN_W//2 - inst.get_width()//2, SCREEN_H - 28))

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