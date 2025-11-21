"""
Simulador AUTOMAX (Mamdani fuzzy) con ciclo día/noche + lluvia + claxon + obstáculo dinámico + rebase animado
Guarda como retro_neon_fuzzy_sim.py
Requisitos: pygame, numpy, scikit-fuzzy, pandas (opcional)
"""

import os
import random
from datetime import datetime

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
ASSETS_DIR = "assets"

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
        for i,_ in enumerate([40,80,140]):
            col = (100, 60+i*40, 200)
            s = pygame.Surface((self.handle_radius*3, self.handle_radius*3), pygame.SRCALPHA)
            pygame.draw.circle(s, col + (40 - i*10,), (self.handle_radius*1, self.handle_radius*1), self.handle_radius + (2-i))
            surf.blit(s, (hx - self.handle_radius -1, line_y - self.handle_radius -1))
        pygame.draw.circle(surf, (255,255,255), (hx, line_y), self.handle_radius)
        lbl = font.render(f"{self.label}: {self.value:.0f}", True, (230,230,230))
        surf.blit(lbl, (x, y - 22))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx,my = event.pos
            x,y,w,h = self.rect
            line_y = y + h//2
            rel = (self.value - self.minv) / (self.maxv - self.minv)
            hx = x + int(rel * w)
            if (mx-hx)**2 + (my-line_y)**2 <= (self.handle_radius+5)**2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
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
        pygame.mixer.init()
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
        self.speed = 40.0        # km/h (slider)
        self.target_speed = 40.0
        self.px_per_m = 5.0

        # coche visual
        self.car_vis_w = 180
        self.car_vis_h = 140
        self.car_x = SCREEN_W // 2 - self.car_vis_w // 2
        self.car_y = SCREEN_H - self.car_vis_h - 18
        self.car_w = self.car_vis_w
        self.car_h = self.car_vis_h

        # assets
        self.try_load_assets()

        # sonidos
        self.rain_sound = None
        self.horn_sound = None
        try:
            rain_path = os.path.join(ASSETS_DIR, "rain_loop.wav.mp3")
            if not os.path.exists(rain_path):
                print(f"ERROR: No se encontró '{rain_path}' en 'assets'. El sonido de lluvia no funcionará.")
            else:
                self.rain_sound = pygame.mixer.Sound(rain_path)
                print("Sonido de lluvia cargado.")
        except Exception as e:
            print(f"Error al cargar rain_loop.wav.mp3: {e}")
        try:
            horn_path = os.path.join(ASSETS_DIR, "horn.wav.mp3")
            if not os.path.exists(horn_path):
                print(f"ERROR: No se encontró '{horn_path}' en 'assets'. El claxon no funcionará.")
            else:
                self.horn_sound = pygame.mixer.Sound(horn_path)
                print("Sonido de claxon cargado.")
        except Exception as e:
            print(f"Error al cargar horn.wav.mp3: {e}")

        # entorno y estado
        self.obst_distance_m = 40.0
        self.visibility = 100.0
        self.visibility_before_rain = 100.0
        self.daytime = 12.0           # hora simulada (0..24)
        self.headlights_on = False
        self.obst_base_x = self.car_x + 400
        self.time = 0.0

        # física
        self.dt = 1.0 / FPS
        self.accel = 0.0

        # sliders
        sx = 40; sw = 380; sy = 60
        self.slider_speed = Slider((sx, sy, sw, 30), 0, 120, self.speed, "Velocidad (km/h)")
        self.slider_dist = Slider((sx, sy+80, sw, 30), 0, 100, self.obst_distance_m, "Distancia (m)")
        self.slider_vis = Slider((sx, sy+160, sw, 30), 0, 100, self.visibility, "Visibilidad (%)")

        # modos y listas
        self.demo_mode = False
        self.demo_timer = 0.0
        self.particles = []
        self.brake_on = False
        self.log = []
        self.road_offset = 0.0
        self.running = True

        # lluvia y claxon
        self.rain_enabled = False
        self.rain_particles = []
        self.horn_playing = False

        # obstáculo dinámico y rebase
        self.obst_speed = 60.0  # km/h fijo
        self.obst_lane = 0      # 0 centro, 1 izquierda, 2 derecha
        self.rebase_anim = 0.0  # progreso 0..1
        self.rebase_dir = 0     # -1 izq, +1 der
        self.rebase_particles = []

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

    # -------- Día/Noche: calcula visibilidad por hora --------
    def day_night_visibility(self):
        h = self.daytime % 24.0
        if 7 <= h <= 19:
            return 100.0  # día
        elif 19 < h <= 22:
            return max(30.0, 100.0 - (h - 19) * 20)  # atardecer 100→40 aprox
        elif 22 < h or h < 6:
            return 30.0   # noche profunda
        elif 6 <= h < 7:
            return 30.0 + (h - 6) * 70  # amanecer 30→100
        return 100.0

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
            # Cerrar ventana
            if e.type == pygame.QUIT:
                self.running = False

            # Sliders
            self.slider_speed.handle_event(e)
            self.slider_dist.handle_event(e)
            self.slider_vis.handle_event(e)

            # Teclado
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_SPACE:
                    self.demo_mode = not self.demo_mode
                    print("Demo mode:", self.demo_mode)
                elif e.key == pygame.K_s:
                    self.save_log_csv()
                    print("Log guardado.")
                elif e.key == pygame.K_r:
                    self.reset_sim()
                elif e.key == pygame.K_l:
                    self.rain_enabled = not self.rain_enabled
                    if self.rain_enabled:
                        self.visibility_before_rain = self.slider_vis.value
                        if self.rain_sound:
                            self.rain_sound.play(loops=-1)
                    else:
                        self.slider_vis.value = self.visibility_before_rain
                        if self.rain_sound:
                            self.rain_sound.stop()
                    print(f"Lluvia activada: {self.rain_enabled}")

            # Mouse: arrastrar obstáculo
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if self.obstacle_screen_rect().collidepoint(mx, my):
                    self.dragging_obstacle = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.dragging_obstacle = False
            elif e.type == pygame.MOUSEMOTION and getattr(self, "dragging_obstacle", False):
                mx, my = e.pos
                SCREEN_Wi, SCREEN_Hi = self.screen.get_size()
                road_horizon_y = 120
                car_front_y = self.car_y
                my_clamped = max(road_horizon_y + 10, min(car_front_y - 20, my))
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
        self.daytime = 12.0
        self.headlights_on = False
        self.rain_enabled = False
        self.horn_playing = False
        self.obst_lane = 0
        self.rebase_anim = 0.0
        self.rebase_particles = []
        if self.rain_sound:
            self.rain_sound.stop()

    def update(self, dt):
        # Lectura sliders
        self.speed = self.slider_speed.value
        # OJO: obst_distance_m se actualizará por dinámica; slider es espejo

        # Avanza ciclo día/noche
        self.daytime = (self.daytime + dt * 0.25) % 24.0
        day_vis = self.day_night_visibility()

        # Visibilidad efectiva: mínimo entre slider y día/noche
        effective_vis = min(self.slider_vis.value, day_vis)

        # Lluvia: fuerza límite adicional
        if self.rain_enabled:
            effective_vis = min(effective_vis, 40.0)
            # Limitar velocidad en lluvia
            max_speed_in_rain = 50.0
            if self.speed > max_speed_in_rain:
                self.slider_speed.value = max(max_speed_in_rain, self.speed - dt * 25.0)
                self.speed = self.slider_speed.value

        # Aplicar visibilidad y luces
        self.visibility = effective_vis
        self.headlights_on = self.visibility < 60.0

        # Modo demo
        if self.demo_mode:
            self.demo_timer += dt
            if self.demo_timer > 2.0:
                self.demo_timer = 0.0
                self.slider_speed.value = random.uniform(20, 100)
                # Distancia será recalculada por dinámica; el slider se actualizará
                if self.rain_enabled:
                    self.slider_vis.value = random.uniform(20, 60)
                else:
                    self.slider_vis.value = random.uniform(50, 100)
                self.speed = self.slider_speed.value

        # --- Fuzzy decision ---
        action_val = self.fuzzy.compute(self.speed, self.obst_distance_m, self.visibility)

        # Map a aceleración
        if action_val < 35:
            accel = -60.0
            self.brake_on = True
            self.add_particles(intensity=1.2, direction=-1)
            action_text = "FRENAR"
        elif action_val > 65:
            accel = 18.0
            self.brake_on = False
            self.add_particles(intensity=1.6, direction=1)
            action_text = "ACELERAR"
        else:
            accel = -6.0
            self.brake_on = False
            action_text = "MANTENER"

        # Claxon automático
        is_close = self.obst_distance_m < 30.0
        if is_close and not self.horn_playing and self.horn_sound:
            self.horn_sound.play(loops=-1)
            self.horn_playing = True
        elif not is_close and self.horn_playing and self.horn_sound:
            self.horn_sound.stop()
            self.horn_playing = False

        # Integración velocidad visual
        if not hasattr(self, 'display_speed'):
            self.display_speed = self.speed
        self.display_speed += accel * dt
        relax = 1.8
        self.display_speed += (self.speed - self.display_speed) * min(1.0, relax * dt)
        self.display_speed = max(0.0, min(120.0, self.display_speed))

        # Movimiento de líneas
        vel_m_s = (self.display_speed * 1000.0) / 3600.0
        px_move = vel_m_s * self.px_per_m * dt
        self.road_offset = (self.road_offset + px_move) % 60

        # --- Dinámica del obstáculo (60 km/h fijo) ---
        car_vel_ms = (self.display_speed * 1000.0) / 3600.0
        obst_vel_ms = (self.obst_speed * 1000.0) / 3600.0
        rel_vel = car_vel_ms - obst_vel_ms   # >0: te acercas; <0: se aleja
        self.obst_distance_m -= rel_vel * dt
        self.obst_distance_m = max(0.0, min(100.0, self.obst_distance_m))
        self.slider_dist.value = self.obst_distance_m

        # --- Rebase/esquive automático y animación ---
        if self.obst_distance_m <= 5.0 and self.rebase_anim <= 0.0:
            # Simula que lo rebasaste: reaparece adelante en carril aleatorio
            self.obst_distance_m = random.uniform(80.0, 100.0)
            self.slider_dist.value = self.obst_distance_m
            self.obst_lane = random.choice([0,1,2])

            # Activar animación de rebase
            self.rebase_anim = 1.0
            self.rebase_dir = random.choice([-1, 1])
            # Generar partículas de velocidad
            self.rebase_particles = []
            cx_center = self.car_x + self.car_w // 2
            cy_center = self.car_y + self.car_h // 2
            for _ in range(30):
                self.rebase_particles.append([
                    cx_center,
                    cy_center,
                    random.uniform(0.5, 1.2),   # vida
                    random.uniform(-200, 200),  # vx
                    random.uniform(-300, -100)  # vy
                ])

        # Partículas normales
        self.update_particles(dt)
        # Actualizar partículas de rebase
        self.update_rebase_particles(dt)
        # Lluvia
        self.update_rain_particles(dt)

        # Log
        self.log.append({
            "time": round(self.time,3),
            "slider_speed": round(self.speed,3),
            "display_speed": round(self.display_speed,3),
            "distance_m": round(self.obst_distance_m,3),
            "visibility": round(self.visibility,3),
            "action_val": round(action_val,3),
            "action_text": action_text,
            "hour": round(self.daytime,2)
        })

    def add_particles(self, intensity=1.0, direction=1):
        llanta_izq_x = self.car_x + 30 + random.uniform(-2, 2)
        llanta_der_x = self.car_x + self.car_w - 50 + random.uniform(-2, 2)
        y = self.car_y + self.car_h - 25 + random.uniform(0, 8)
        for _ in range(int(1 + intensity)):
            life = random.uniform(0.4, 1.0)
            vx = random.uniform(10,60) * (0.02 * (-direction))
            vy = random.uniform(-10,5) * 0.02
            self.particles.append([llanta_izq_x, y, life, vx, vy])
        for _ in range(int(1 + intensity)):
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

    def update_rebase_particles(self, dt):
        newp = []
        for p in self.rebase_particles:
            x,y,life,vx,vy = p
            life -= dt
            x += vx * dt
            y += vy * dt
            if life > 0:
                newp.append([x,y,life,vx,vy])
        self.rebase_particles = newp

    # -------- Lluvia --------
    def update_rain_particles(self, dt):
        if not self.rain_enabled:
            self.rain_particles = []
            return
        if random.random() < 0.8:
            for _ in range(15):
                x = random.randint(0, SCREEN_W)
                y = random.randint(-50, -10)
                l = random.randint(15, 25)
                speed = random.randint(15, 20)
                self.rain_particles.append([x, y, l, speed])
        new_rain = []
        for p in self.rain_particles:
            p[1] += p[3]
            if p[1] < SCREEN_H:
                new_rain.append(p)
        self.rain_particles = new_rain

    def draw_rain_particles(self, surf):
        if not self.rain_enabled:
            return
        for x, y, length, _ in self.rain_particles:
            pygame.draw.line(surf, (150, 180, 255), (x, y), (x, y + length), 1)

    def obstacle_screen_rect(self):
        SCREEN_Wi, SCREEN_Hi = self.screen.get_size()
        center_x = SCREEN_Wi // 2
        road_horizon_y = 120
        car_front_y = self.car_y
        d = max(0.0, min(100.0, self.obst_distance_m))
        t = d / 100.0
        max_travel = max(20.0, (car_front_y - road_horizon_y - 40.0))
        y = int(road_horizon_y + (1.0 - t) * max_travel)
        width = int(max(24, 24 + (1.0 - t) * 120))
        height = int(max(20, 20 + (1.0 - t) * 80))
        # Ajuste de carril
        lane_offset = 0
        if self.obst_lane == 1:
            lane_offset = -80
        elif self.obst_lane == 2:
            lane_offset = 80
        x = center_x - width // 2 + lane_offset
        return pygame.Rect(x, y, width, height)

    def draw_neon_text(self, surf, text, pos, size=24, glow_color=(120,60,220)):
        f = pygame.font.SysFont("Consolas", size, bold=True)
        base = f.render(text, True, (255,255,255))
        x,y = pos
        for i,_ in enumerate([60,30,15]):
            s = f.render(text, True, (glow_color[0], glow_color[1], glow_color[2]))
            s.set_alpha(80 - i*20)
            surf.blit(s, (x- i -2, y - i -2))
        surf.blit(base, pos)

    def draw(self):
        s = self.screen
        SCREEN_Wi, SCREEN_Hi = s.get_size()
        center_x = SCREEN_Wi // 2

        # --- Fondo dinámico SOLO por hora (día/noche) ---
        h = self.daytime % 24.0
        if 7 <= h <= 19:
            day_factor = 1.0            # día
        elif 19 < h <= 22:
            day_factor = max(0.0, 1.0 - (h - 19) / 3.0)  # atardecer
        elif 22 < h or h < 6:
            day_factor = 0.0            # noche
        else:  # 6 <= h < 7
            day_factor = (h - 6)        # amanecer
        top_color = int(40 + 120 * day_factor)   # 40 noche → 160 día
        bot_color = int(60 + 160 * day_factor)   # 60 noche → 220 día
        for i in range(SCREEN_Hi):
            t = i / SCREEN_Hi
            r = int((top_color * (1 - t) + (bot_color - 30) * t))
            g = int((top_color * 0.9 * (1 - t) + (bot_color - 40) * t))
            b = int((top_color * 1.4 * (1 - t) + (bot_color + 50) * t))
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            pygame.draw.line(s, (r, g, b), (0, i), (SCREEN_Wi, i))

        # Oscurecer si llueve (overlay)
        if self.rain_enabled:
            dark_overlay = pygame.Surface((SCREEN_Wi, SCREEN_Hi), pygame.SRCALPHA)
            dark_overlay.fill((0, 0, 10, 90))
            s.blit(dark_overlay, (0, 0))

        # --- Carretera ---
        road_bottom_y = SCREEN_Hi
        road_horizon_y = 110
        lane_color = (70, 170, 255)
        asphalt_color = (20, 20, 24)
        road_left_w = 180
        road_right_w = 180
        road_horizon_inner = 40
        pygame.draw.polygon(s, asphalt_color, [
            (center_x - road_left_w, road_bottom_y),
            (center_x - road_horizon_inner, road_horizon_y),
            (center_x + road_horizon_inner, road_horizon_y),
            (center_x + road_right_w, road_bottom_y)
        ])
        pygame.draw.line(s, lane_color, (center_x - road_left_w, road_bottom_y), (center_x - road_horizon_inner, road_horizon_y), 3)
        pygame.draw.line(s, lane_color, (center_x + road_left_w, road_bottom_y), (center_x + road_horizon_inner, road_horizon_y), 3)

        # --- Líneas centrales animadas ---
        dash_h = 40
        gap = 90
        vel_m_s = (getattr(self, "display_speed", 0.0) * 1000.0) / 3600.0
        move_px = vel_m_s * self.px_per_m * self.dt * 2.2
        self.road_offset = (self.road_offset + move_px) % (dash_h + gap)
        offset_pix = int(self.road_offset)
        lane_x = center_x - 2
        total_height = SCREEN_Hi - road_horizon_y
        for y in range(int(road_horizon_y - (dash_h + gap)),
                       int(SCREEN_Hi + (dash_h + gap)),
                       dash_h + gap):
            yy = y + offset_pix
            if yy < road_horizon_y - dash_h or yy > SCREEN_Hi + dash_h:
                continue
            t = (yy - road_horizon_y) / total_height
            scale = 1.0 - min(1.0, t * 0.9)
            dash_w = int(3 * (1.0 + scale * 1.3))
            dash_h_scaled = int(dash_h * (0.5 + scale * 0.7))
            if yy < road_horizon_y + 100:
                alpha = int(255 * ((yy - road_horizon_y) / 100.0))
            elif yy > SCREEN_Hi - 120:
                alpha = int(255 * ((SCREEN_Hi - yy) / 120.0))
            else:
                alpha = 255
            alpha = max(0, min(255, alpha))
            brightness = int(180 + min(60, self.display_speed * 0.6))
            line_color = (brightness, brightness, 255)
            surf = pygame.Surface((dash_w, dash_h_scaled), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*line_color, alpha), (0, 0, dash_w, dash_h_scaled))
            s.blit(surf, (lane_x - dash_w // 2, yy - dash_h_scaled // 2))

        # --- Obstáculo ---
        rect = self.obstacle_screen_rect()
        if self.obst_sprite:
            sprite = pygame.transform.smoothscale(self.obst_sprite, (rect.w, rect.h))
            s.blit(sprite, (rect.x, rect.y))
        else:
            glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
            pygame.draw.rect(glow, (215,75,75,60), (0,0,rect.w+16, rect.h+16), border_radius=6)
            s.blit(glow, (rect.x-8, rect.y-8))
            pygame.gfxdraw.box(s, rect, (215,75,75))
            pygame.gfxdraw.rectangle(s, rect, (255,120,120))

        # --- Lluvia detrás del coche ---
        self.draw_rain_particles(s)

        # --- Coche (con animación de rebase) ---
        cx, cy, cw, ch = self.car_x, self.car_y, self.car_w, self.car_h
        if self.rebase_anim > 0.0:
            offset_x = int(self.rebase_dir * 80 * self.rebase_anim)
            offset_y = int(-40 * self.rebase_anim)
            cx += offset_x
            cy += offset_y
            self.rebase_anim = max(0.0, self.rebase_anim - 0.05)

            # Overlay translúcido (sensación de desenfoque)
            blur = pygame.Surface((SCREEN_Wi, SCREEN_Hi), pygame.SRCALPHA)
            blur.fill((255, 255, 200, 40))
            s.blit(blur, (0, 0))

        if self.car_sprite:
            car_img = pygame.transform.smoothscale(self.car_sprite, (cw, ch))
            s.blit(car_img, (cx, cy))
        else:
            body_color = (38, 58, 80)
            pygame.draw.rect(s, body_color, (cx + 8, cy + 8, cw - 16, ch - 16), border_radius=14)
            pygame.draw.rect(s, (70,90,120), (cx + 18, cy + 16, cw - 36, ch // 3), border_radius=8)
            pygame.draw.rect(s, (150, 30, 30), (cx + 40, cy + ch - 40, 10, 1))
            pygame.draw.rect(s, (150, 30, 30), (cx + cw - 60, cy + ch - 40, 10, 1))
            pygame.draw.rect(s, (110,140,190), (cx + cw//2 - 8, cy + ch - 28, 16, 4), border_radius=2)
            pygame.draw.rect(s, (20, 20, 28), (cx, cy + 8, 8, ch - 16))
            pygame.draw.rect(s, (20, 20, 28), (cx + cw - 8, cy + 8, 8, ch - 16))

        # Luces delanteras automáticas
        if self.headlights_on:
            lx = cx + cw//2
            ly = cy + 20
            beam = pygame.Surface((SCREEN_Wi, SCREEN_Hi), pygame.SRCALPHA)
            pygame.draw.polygon(beam, (255, 255, 200, 90),
                [(lx-40, ly), (lx+40, ly), (lx+260, ly-120), (lx-260, ly-120)])
            s.blit(beam, (0,0))

        # Luces de freno
        if getattr(self, "brake_on", False):
            pygame.draw.rect(s, (255,12,12), (cx + 40, cy + ch - 74, 16, 6))
            pygame.draw.rect(s, (255,12,12), (cx + cw - 60, cy + ch - 74, 16, 6))

        # Partículas rebase (destellos)
        for x,y,life,_,_ in self.rebase_particles:
            alpha = int(255 * (life / 1.2))
            surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255,200,80, alpha), (2,2), 2)
            s.blit(surf, (int(x), int(y)))

        # Partículas normales (humo/freno)
        for x,y,life,_,_ in self.particles:
            alpha = int(255 * (life/1.0))
            surf = pygame.Surface((6,6), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255,180,60, alpha), (3,3), 3)
            s.blit(surf, (x,y))

        # HUD Izquierdo
        hud = pygame.Surface((380, 340), pygame.SRCALPHA)
        hud.fill((10,10,10,160))
        s.blit(hud, (20, 30))
        self.slider_speed.draw(s, self.font)
        self.slider_dist.draw(s, self.font)
        self.slider_vis.draw(s, self.font)
        hour_txt = self.font.render(f"Hora: {int(self.daytime):02d}:{int((self.daytime%1)*60):02d}", True, (210,210,210))
        s.blit(hour_txt, (30, 30 + 300))

        # HUD Central (velocidad)
        speed_text = self.bigfont.render(f"{getattr(self, 'display_speed', 0.0):.0f}", True, (255,255,255))
        unit_text = self.font.render("km/h", True, (190,190,190))
        s.blit(speed_text, (center_x - 20, 28))
        s.blit(unit_text, (center_x + 46, 56))

        # Barras laterales (Visibilidad/Distancia)
        vis_bar_h = 120
        dist_bar_h = 120
        bar_w = 16
        base_y = SCREEN_Hi - vis_bar_h - 80
        base_x = SCREEN_Wi - 100

        # Visibilidad
        vis_x = base_x
        vis_y = base_y
        pygame.draw.rect(s, (40, 40, 40), (vis_x, vis_y, bar_w, vis_bar_h), border_radius=4)
        vis_fill_h = int((self.visibility / 100.0) * vis_bar_h)
        grad_surf = pygame.Surface((bar_w, vis_fill_h))
        for yy in range(vis_fill_h):
            tt = yy / max(1, vis_fill_h - 1)
            r = int(220 * (1 - tt) + 60 * tt)
            g = int(60 * (1 - tt) + 200 * tt)
            b = int(60 * (1 - tt) + 80 * tt)
            pygame.draw.line(grad_surf, (r, g, b), (0, vis_fill_h - 1 - yy), (bar_w, vis_fill_h - 1 - yy))
        s.blit(grad_surf, (vis_x, vis_y + (vis_bar_h - vis_fill_h)))
        text_vis = self.font.render("Vis", True, (200, 200, 200))
        s.blit(text_vis, (vis_x - 4, vis_y + vis_bar_h + 6))

        # Distancia
        dist_x = base_x + 40
        dist_y = base_y
        pygame.draw.rect(s, (40, 40, 40), (dist_x, dist_y, bar_w, dist_bar_h), border_radius=4)
        dnorm = max(0.0, min(1.0, 1.0 - (self.obst_distance_m / 100.0)))
        dist_fill_h = int(dnorm * dist_bar_h)
        dist_surf = pygame.Surface((bar_w, dist_fill_h))
        for yy in range(dist_fill_h):
            tt = yy / max(1, dist_fill_h - 1)
            r = int(80 * (1 - tt) + 220 * tt)
            g = int(180 * (1 - tt) + 60 * tt)
            b = int(60 * (1 - tt) + 70 * tt)
            pygame.draw.line(dist_surf, (r, g, b), (0, dist_fill_h - 1 - yy), (bar_w, dist_fill_h - 1 - yy))
        s.blit(dist_surf, (dist_x, dist_y + (dist_bar_h - dist_fill_h)))
        text_dist = self.font.render("Dist", True, (200, 200, 200))
        s.blit(text_dist, (dist_x - 8, dist_y + dist_bar_h + 6))

        # Estado derecha
        if self.log:
            last = self.log[-1]
            act = last["action_text"]
            action_val = last["action_val"]
            txt = f"Acción: {act} ({action_val:.1f})"
            actsurf = self.font.render(txt, True, (255, 200, 120))
            s.blit(actsurf, (SCREEN_Wi - 340, 48))

        # Instrucciones
        inst = self.font.render("L: Lluvia | SPACE: Demo | S: Guardar CSV | R: Reset | ESC: Salir", True, (180,180,180))
        s.blit(inst, (SCREEN_Wi//2 - inst.get_width()//2, SCREEN_Hi - 28))

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
