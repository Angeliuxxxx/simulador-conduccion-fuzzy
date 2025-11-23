"""
Simulador AUTOMAX ULTIMATE (Versión Arcade Rápida + Visuales Full)
- Movimiento: 100% Responsivo al Slider (Sin lag, sin frenos fantasmas).
- Visuales: Luces rojas encima del coche, Chispas al frenar, Niebla densa.
- UI: Slider de visibilidad regresa solo en modo Soleado.
"""

import os
import sys
import math
import random
from datetime import datetime
import numpy as np
import pandas as pd
import pygame
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Configuración ---
SCREEN_W = 1200
SCREEN_H = 600
FPS = 60
ASSETS_DIR = "assets"

# ==========================================
#  CEREBRO DIFUSO (Solo para feedback visual)
# ==========================================
class FuzzyController:
    def __init__(self):
        # Antecedentes
        self.vel = ctrl.Antecedent(np.arange(0, 121, 1), 'velocidad')
        self.dist = ctrl.Antecedent(np.arange(0, 101, 1), 'distancia')
        self.vis = ctrl.Antecedent(np.arange(0, 101, 1), 'visibilidad')
        self.grip = ctrl.Antecedent(np.arange(0, 101, 1), 'adherencia')

        # Consecuentes
        self.brake = ctrl.Consequent(np.arange(0, 101, 1), 'freno')
        self.throttle = ctrl.Consequent(np.arange(0, 101, 1), 'acelerador')
        self.horn = ctrl.Consequent(np.arange(0, 101, 1), 'claxon')

        # Membresías
        self.vel['baja'] = fuzz.trimf(self.vel.universe, [0, 0, 50])
        self.vel['media'] = fuzz.trimf(self.vel.universe, [30, 60, 90])
        self.vel['alta'] = fuzz.trimf(self.vel.universe, [70, 120, 120])

        self.dist['corta'] = fuzz.trimf(self.dist.universe, [0, 0, 30])
        self.dist['media'] = fuzz.trimf(self.dist.universe, [20, 50, 80])
        self.dist['larga'] = fuzz.trimf(self.dist.universe, [60, 100, 100])

        self.vis['baja'] = fuzz.trimf(self.vis.universe, [0, 0, 40])
        self.vis['media'] = fuzz.trimf(self.vis.universe, [30, 60, 90])
        self.vis['alta'] = fuzz.trimf(self.vis.universe, [70, 100, 100])

        self.grip['resbaloso'] = fuzz.trimf(self.grip.universe, [0, 0, 50])
        self.grip['normal'] = fuzz.trimf(self.grip.universe, [40, 100, 100])

        self.brake['nada'] = fuzz.trimf(self.brake.universe, [0, 0, 10])
        self.brake['suave'] = fuzz.trimf(self.brake.universe, [10, 40, 70])
        self.brake['fuerte'] = fuzz.trimf(self.brake.universe, [50, 100, 100])

        self.throttle['nada'] = fuzz.trimf(self.throttle.universe, [0, 0, 10])
        self.throttle['crucero'] = fuzz.trimf(self.throttle.universe, [10, 40, 70])
        self.throttle['fondo'] = fuzz.trimf(self.throttle.universe, [50, 100, 100])

        self.horn['silencio'] = fuzz.trimf(self.horn.universe, [0, 0, 50])
        self.horn['alerta'] = fuzz.trimf(self.horn.universe, [40, 100, 100])

        # Reglas
        rules = []
        rules.append(ctrl.Rule(self.dist['corta'], [self.brake['fuerte'], self.throttle['nada'], self.horn['alerta']]))
        rules.append(ctrl.Rule(self.grip['resbaloso'] & self.vel['alta'], [self.brake['suave'], self.throttle['nada'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.vis['baja'] & self.vel['alta'], [self.brake['suave'], self.throttle['nada'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.dist['media'] & self.vel['alta'], [self.brake['suave'], self.throttle['nada'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.dist['media'] & self.vel['media'], [self.brake['nada'], self.throttle['crucero'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.dist['media'] & self.vel['baja'], [self.brake['nada'], self.throttle['fondo'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.dist['larga'] & self.vis['alta'] & self.grip['normal'], [self.brake['nada'], self.throttle['fondo'], self.horn['silencio']]))
        rules.append(ctrl.Rule(self.dist['larga'] & self.vis['baja'], [self.brake['nada'], self.throttle['crucero'], self.horn['silencio']]))

        self.ctrl_sys = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_sys)

    def compute(self, v, d, vis, g):
        self.sim.input['velocidad'] = np.clip(v, 0, 120)
        self.sim.input['distancia'] = np.clip(d, 0, 100)
        self.sim.input['visibilidad'] = np.clip(vis, 0, 100)
        self.sim.input['adherencia'] = np.clip(g, 0, 100)
        try:
            self.sim.compute()
            return self.sim.output['freno'], self.sim.output['acelerador'], self.sim.output['claxon']
        except:
            return 0, 0, 0

# ==========================================
#  UI
# ==========================================
class Slider:
    def __init__(self, rect, minv, maxv, val, label):
        self.rect = pygame.Rect(rect)
        self.minv, self.maxv, self.value = minv, maxv, val
        self.label = label
        self.dragging = False

    def draw(self, surf, font):
        pygame.draw.rect(surf, (50,50,50), (self.rect.x, self.rect.y + 12, self.rect.w, 6), border_radius=3)
        pct = (self.value - self.minv) / (self.maxv - self.minv)
        hx = self.rect.x + int(pct * self.rect.w)
        for i,_ in enumerate([40,80]):
            pygame.draw.circle(surf, (100, 60+i*40, 200, 100), (hx, self.rect.y+15), 14-i*2)
        pygame.draw.circle(surf, (255,255,255), (hx, self.rect.y + 15), 10)
        lbl = font.render(f"{self.label}: {self.value:.0f}", True, (220,220,220))
        surf.blit(lbl, (self.rect.x, self.rect.y - 20))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self.rect.collidepoint(mx, my-15) or self.rect.collidepoint(mx, my+15): 
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = event.pos
            rel = (mx - self.rect.x) / self.rect.w
            val = self.minv + rel * (self.maxv - self.minv)
            self.value = max(self.minv, min(self.maxv, val))

class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = (70, 70, 90)
        self.hover_color = (100, 100, 120)

    def draw(self, surf, font):
        mouse_pos = pygame.mouse.get_pos()
        col = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        pygame.draw.rect(surf, (200, 200, 200), self.rect, 2, border_radius=5)
        txt_surf = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surf.blit(txt_surf, txt_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# ==========================================
#  SIMULACIÓN PRINCIPAL
# ==========================================
class RetroNeonSim:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("Automax Ultimate - Final Version")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.bigfont = pygame.font.SysFont("Consolas", 28, bold=True)

        # Coche
        self.car_vis_w = 180; self.car_vis_h = 140
        self.car_x = SCREEN_W // 2 - 90; self.car_y = SCREEN_H - 160
        self.car_w = 180; self.car_h = 140
        
        self.load_assets()
        self.fuzzy = FuzzyController()

        # Estado Inicial (Importante: display_speed)
        self.speed = 40.0
        self.display_speed = 40.0 
        self.px_per_m = 5.0 
        self.obst_distance_m = 40.0
        self.visibility = 100.0
        self.grip = 100.0
        self.daytime = 12.0
        self.headlights_on = False
        
        # Juego
        self.obst_speed = 60.0 
        self.rebase_anim = 0.0
        self.rebase_dir = 0
        self.obst_lane = 0 

        # Sliders
        self.slider_speed = Slider((40, 60, 380, 30), 0, 120, 40, "Velocidad (km/h)")
        self.slider_dist = Slider((40, 140, 380, 30), 0, 100, 40, "Distancia (m)")
        self.slider_vis = Slider((40, 220, 380, 30), 0, 100, 100, "Visibilidad (%)")

        # Botones
        self.btn_time = Button(40, 320, 160, 40, "CAMBIAR HORA")
        self.btn_weather = Button(220, 320, 180, 40, "CAMBIAR CLIMA")

        # Clima
        self.weather_mode = 0 
        self.weather_names = ["SOLEADO", "LLUVIA", "NEBLINA", "POLVO"]
        self.rain_enabled = False
        self.horn_playing = False
        self.brake_val = 0.0
        self.throttle_val = 0.0
        self.action_text = "MANTENIENDO"
        
        self.particles = []
        self.rain_particles = []
        self.rebase_particles = []
        self.log = []
        self.road_offset = 0.0
        self.time = 0.0
        self.demo_mode = False
        self.demo_timer = 0.0
        self.running = True
        
        # Texturas
        self.cloud_surf = self.generate_cloud_texture((SCREEN_W + 200, SCREEN_H), (255,255,255), 200)
        self.dust_surf = self.generate_cloud_texture((SCREEN_W + 200, SCREEN_H), (160, 110, 50), 250)
        self.cloud_offset_x = 0.0

    def load_assets(self):
        self.car_sprite = None; self.obst_sprite = None
        self.rain_sound = None; self.horn_sound = None
        try:
            if os.path.exists(f"{ASSETS_DIR}/car.png"):
                self.car_sprite = pygame.transform.smoothscale(pygame.image.load(f"{ASSETS_DIR}/car.png"), (self.car_w, self.car_h))
            if os.path.exists(f"{ASSETS_DIR}/obstacle.png"):
                self.obst_sprite = pygame.image.load(f"{ASSETS_DIR}/obstacle.png")
            
            r_files = ["rain_loop.wav.mp3", "rain_loop.wav"]
            for f in r_files:
                if os.path.exists(f"{ASSETS_DIR}/{f}"):
                    self.rain_sound = pygame.mixer.Sound(f"{ASSETS_DIR}/{f}"); self.rain_sound.set_volume(0.5); break
            
            h_files = ["horn.wav.mp3", "horn.wav"]
            for f in h_files:
                if os.path.exists(f"{ASSETS_DIR}/{f}"):
                    self.horn_sound = pygame.mixer.Sound(f"{ASSETS_DIR}/{f}"); break
        except Exception as e: print(f"Assets error: {e}")

    def generate_cloud_texture(self, size, color, density):
        surf = pygame.Surface(size, pygame.SRCALPHA)
        for _ in range(density):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            radius = random.randint(80, 200)
            alpha = random.randint(40, 100) 
            for i in range(radius, 0, -20):
                a = int(alpha * (i/radius))
                pygame.draw.circle(surf, (*color, a), (x, y), i)
        return surf

    def day_night_visibility(self):
        h = self.daytime % 24.0
        if 7 <= h <= 19: return 100.0
        elif 19 < h <= 22: return max(30.0, 100.0 - (h - 19) * 20)
        elif 22 < h or h < 6: return 30.0
        elif 6 <= h < 7: return 30.0 + (h - 6) * 70
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
            if e.type == pygame.QUIT: self.running = False
            
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: self.running = False
                elif e.key == pygame.K_SPACE: self.demo_mode = not self.demo_mode
                elif e.key == pygame.K_s: self.save_log_csv()
                elif e.key == pygame.K_r: self.reset_sim()
                elif e.key == pygame.K_c or e.key == pygame.K_w: self.cycle_weather()

            if self.btn_time.is_clicked(e):
                if 6 <= self.daytime <= 18: self.daytime = 0.0
                else: self.daytime = 12.0
            
            if self.btn_weather.is_clicked(e):
                self.cycle_weather()

            self.slider_speed.handle_event(e)
            self.slider_dist.handle_event(e)
            self.slider_vis.handle_event(e)

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if self.obstacle_screen_rect().collidepoint(e.pos): self.dragging_obstacle = True
            elif e.type == pygame.MOUSEBUTTONUP: self.dragging_obstacle = False
            elif e.type == pygame.MOUSEMOTION and getattr(self, "dragging_obstacle", False):
                mx, my = e.pos
                t = 1.0 - ((my - 120) / (self.car_y - 160))
                self.obst_distance_m = max(0, min(100, t * 100))
                self.slider_dist.value = self.obst_distance_m

    def cycle_weather(self):
        self.weather_mode = (self.weather_mode + 1) % 4
        if self.weather_mode == 1 and self.rain_sound: self.rain_sound.play(-1)
        elif self.rain_sound: self.rain_sound.stop()

    def update(self, dt):
        self.speed = self.slider_speed.value
        self.daytime = (self.daytime + dt * 0.05) % 24.0
        
        # --- LÍMITES POR CLIMA ---
        day_vis = self.day_night_visibility()
        target_vis = 100; self.grip = 100
        self.rain_enabled = False; self.fog_enabled = False; self.fog_color=(200,200,200)

        if self.weather_mode == 1: # Lluvia
            target_vis = 40; self.grip = 40; self.rain_enabled = True
        elif self.weather_mode == 2: # Niebla
            target_vis = 25; self.grip = 90; self.fog_enabled = True; self.fog_color = (220,220,230)
        elif self.weather_mode == 3: # Polvo
            target_vis = 50; self.grip = 70; self.fog_enabled = True; self.fog_color = (160,120,50)

        final_vis = min(self.slider_vis.value, day_vis, target_vis)
        
        # --- LOGICA DE RETORNO AUTOMÁTICO SLIDER ---
        # Si el clima es SOLEADO (0) y el slider está abajo, que suba solo
        if self.weather_mode == 0 and not self.slider_vis.dragging:
            if self.slider_vis.value < day_vis:
                self.slider_vis.value += (day_vis - self.slider_vis.value) * 0.05
        # Si el clima es MALO, que baje solo
        elif not self.slider_vis.dragging and self.slider_vis.value > final_vis + 1:
             self.slider_vis.value += (final_vis - self.slider_vis.value) * 0.1

        self.visibility = self.slider_vis.value
        
        self.headlights_on = (self.visibility < 60 or self.weather_mode != 0 or (self.daytime < 6 or self.daytime > 19))

        if self.demo_mode:
            self.demo_timer += dt
            if self.demo_timer > 3:
                self.demo_timer = 0
                self.slider_speed.value = random.uniform(30, 100)
                if random.random()<0.2: self.cycle_weather()

        # FUZZY
        brake, throttle, horn = self.fuzzy.compute(self.display_speed, self.obst_distance_m, self.visibility, self.grip)
        self.brake_val = brake
        self.throttle_val = throttle

        # --- FÍSICA 100% RESPONSIVA (El slider es ley) ---
        target_speed = self.speed # La que dice el slider
        
        # Simplemente nos movemos suavemente hacia la velocidad del slider
        diff = target_speed - self.display_speed
        self.display_speed += diff * dt * 5.0 
        self.display_speed = max(0, min(120, self.display_speed))
        
        # OBSTÁCULO Y REBASE
        rel_kmh = self.display_speed - self.obst_speed
        rel_ms = (rel_kmh * 1000) / 3600
        
        if not getattr(self, "dragging_obstacle", False):
            self.obst_distance_m -= rel_ms * dt * 0.8 
            
            if self.obst_distance_m <= 0.0:
                if self.rebase_anim <= 0:
                    self.rebase_anim = 1.0
                    self.rebase_dir = random.choice([-1, 1])
                    self.obst_distance_m = 100.0
                    self.obst_lane = random.choice([-1, 0, 1])
                    self.gen_rebase_particles()
            
            self.obst_distance_m = max(0, min(100, self.obst_distance_m))
            self.slider_dist.value = self.obst_distance_m

        if horn > 60:
            if not self.horn_playing and self.horn_sound: self.horn_sound.play(-1); self.horn_playing=True
        else:
            if self.horn_playing and self.horn_sound: self.horn_sound.stop(); self.horn_playing=False

        # Texto Acción
        if brake > throttle + 10: self.action_text = "FRENAR"
        elif throttle > brake + 10: self.action_text = "ACELERANDO"
        else: self.action_text = "MANTENIENDO"

        vel_ms = self.display_speed / 3.6
        self.road_offset = (self.road_offset + vel_ms * 5 * dt) % 60
        self.cloud_offset_x = (self.cloud_offset_x + vel_ms * 0.8 * dt) % 200
        
        self.update_particles(dt)
        self.update_rebase_particles(dt)
        self.update_rain_particles(dt)
        self.log.append({"t":round(self.time,2), "v":round(self.display_speed,1)})

    def gen_rebase_particles(self):
        cx, cy = self.car_x + self.car_w//2, self.car_y + self.car_h//2
        for _ in range(20): self.rebase_particles.append([cx, cy, random.uniform(0.5,1.2), random.uniform(-200,200), random.uniform(-300,-100)])

    def update_particles(self, dt):
        # Generar Chispas (Si frena muy fuerte) o Humo (Freno normal)
        if self.brake_val > 10:
            lx = self.car_x + 40; rx = self.car_x + self.car_w - 60; y = self.car_y + self.car_h - 10
            
            color = (200, 200, 200) # Humo
            if self.brake_val > 60: color = (255, 150, 50) # Chispas naranjas
            
            self.particles.append([lx+random.uniform(-5,5), y, 0.5, random.uniform(-5,5), 10, color])
            self.particles.append([rx+random.uniform(-5,5), y, 0.5, random.uniform(-5,5), 10, color])
        
        new_p = []
        for p in self.particles:
            p[2] -= dt; p[0]+=p[3]*dt; p[1]+=p[4]*dt
            if p[2]>0: new_p.append(p)
        self.particles = new_p

    def update_rebase_particles(self, dt):
        new_p = []
        for p in self.rebase_particles:
            p[2]-=dt; p[0]+=p[3]*dt; p[1]+=p[4]*dt
            if p[2]>0: new_p.append(p)
        self.rebase_particles = new_p

    def update_rain_particles(self, dt):
        if not self.rain_enabled: self.rain_particles = []; return
        if random.random() < 0.8:
            for _ in range(15): self.rain_particles.append([random.randint(0,SCREEN_W), random.randint(-50,-10), random.randint(15,25), random.randint(15,20)])
        new_r = []
        for p in self.rain_particles:
            p[1]+=p[3]
            if p[1]<SCREEN_H: new_r.append(p)
        self.rain_particles = new_r
    
    def add_particles(self, intensity, direction):
        pass

    def obstacle_screen_rect(self):
        cx = SCREEN_W // 2
        horizon_y = 120
        prog = 1.0 - (self.obst_distance_m / 100)
        y = horizon_y + prog * (self.car_y - horizon_y - 50)
        w = int(100 * (0.2 + 0.8 * prog)); h = int(80 * (0.2 + 0.8 * prog))
        lx = 0
        if self.obst_lane == -1: lx = -80 * prog
        if self.obst_lane == 1: lx = 80 * prog
        return pygame.Rect(cx - w//2 + int(lx), int(y), w, h)

    def draw(self):
        s = self.screen
        cx = SCREEN_W // 2

        # Fondo
        h = self.daytime % 24.0
        if 7<=h<=19: df=1.0
        elif 19<h<=22: df=max(0.0, 1.0-(h-19)/3.0)
        elif 22<h or h<6: df=0.0
        else: df=(h-6)
        top_c = int(40 + 120*df); bot_c = int(60 + 160*df)
        for i in range(SCREEN_H):
            t=i/SCREEN_H
            r=int(top_c*(1-t)+(bot_c-30)*t); g=int(top_c*0.9*(1-t)+(bot_c-40)*t); b=int(top_c*1.4*(1-t)+(bot_c+50)*t)
            pygame.draw.line(s, (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))), (0,i), (SCREEN_W,i))

        # Niebla/Polvo
        if self.fog_enabled:
            fog_layer = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            fog_layer.blit(self.cloud_surf, (-int(self.cloud_offset_x), 0))
            fog_layer.blit(self.cloud_surf, (SCREEN_W - int(self.cloud_offset_x), 0))
            col_layer = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            col_layer.fill((*self.fog_color, 200)) 
            fog_layer.blit(col_layer, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
            s.blit(fog_layer, (0,0))

        # Lluvia Oscuridad
        if self.rain_enabled:
            ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            ov.fill((0,0,10,120))
            s.blit(ov, (0,0))

        # Carretera
        ry = SCREEN_H; hy = 110
        pygame.draw.polygon(s, (20,20,24), [(cx-180, ry), (cx-40, hy), (cx+40, hy), (cx+180, ry)])
        pygame.draw.line(s, (70,170,255), (cx-180, ry), (cx-40, hy), 3)
        pygame.draw.line(s, (70,170,255), (cx+180, ry), (cx+40, hy), 3)

        # Líneas
        off = int(self.road_offset)
        for y in range(int(hy-130), int(ry+130), 130):
            yy = y + off
            if yy < hy or yy > ry: continue
            t = (yy-hy)/(ry-hy)
            w = int(3*(1.0+t*1.3)); h_l = int(40*(0.5+t*0.7))
            pygame.draw.rect(s, (255,255,0), (cx-w//2, yy, w, h_l))

        # Obstáculo
        rect = self.obstacle_screen_rect()
        if self.obst_sprite: s.blit(pygame.transform.smoothscale(self.obst_sprite, (rect.w, rect.h)), rect)
        else: pygame.draw.rect(s, (255,50,50), rect)

        if self.rain_enabled:
            for x,y,l,_ in self.rain_particles: pygame.draw.line(s, (150,180,255), (x,y), (x,y+l), 1)

        cx_car = self.car_x
        if self.rebase_anim > 0.0:
            cx_car += int(self.rebase_dir * 80 * self.rebase_anim)
            self.rebase_anim = max(0.0, self.rebase_anim - 0.05)
            blur = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA); blur.fill((255,255,255,40)); s.blit(blur,(0,0))

        if self.car_sprite: s.blit(self.car_sprite, (cx_car, self.car_y))
        else: pygame.draw.rect(s, (255,255,255), (cx_car, self.car_y, self.car_w, self.car_h))

        # Luces
        if self.headlights_on:
            beam = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.polygon(beam, (255,255,200,60), [(cx_car+20, self.car_y+40), (cx_car+self.car_w-20, self.car_y+40), (cx+300, 0), (cx-300, 0)])
            s.blit(beam, (0,0))
            
        # Luces de freno (DIBUJADAS AL FINAL Y CON COLOR INTENSO)
        if getattr(self, "brake_on", False): # Usar la variable que activa el freno
            # Luz fuerte
            pygame.draw.rect(s, (255, 0, 0), (cx_car+40, self.car_y+100, 20, 10))
            pygame.draw.rect(s, (255, 0, 0), (cx_car+self.car_w-60, self.car_y+100, 20, 10))
            # Resplandor extra
            glow = pygame.Surface((24, 14), pygame.SRCALPHA)
            glow.fill((255, 50, 50, 150))
            s.blit(glow, (cx_car+38, self.car_y+98))
            s.blit(glow, (cx_car+self.car_w-62, self.car_y+98))

        # Partículas
        for p in self.particles:
            alpha = int(255 * p[2])
            surf = pygame.Surface((10, 10), pygame.SRCALPHA)
            col = p[5] if len(p)>5 else (200,200,200)
            pygame.draw.circle(surf, (*col, alpha), (5,5), 5)
            s.blit(surf, (int(p[0])-5, int(p[1])-5))
        
        for p in self.rebase_particles:
            alpha = int(255 * (p[2]/1.2))
            surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255,200,0, alpha), (2,2), 2)
            s.blit(surf, (int(p[0])-2, int(p[1])-2))

        # UI
        panel = pygame.Surface((440, 400), pygame.SRCALPHA)
        panel.fill((80, 70, 100, 220))
        s.blit(panel, (20, 20))
        
        self.slider_speed.draw(s, self.font)
        self.slider_dist.draw(s, self.font)
        self.slider_vis.draw(s, self.font)
        
        # Botones
        self.btn_time.draw(s, self.font)
        self.btn_weather.draw(s, self.font)

        s.blit(self.font.render(f"Hora: {int(self.daytime):02d}:00", True, (255,255,255)), (40, 280))
        w_c = (255,255,0)
        s.blit(self.bigfont.render(f"CLIMA: {self.weather_names[self.weather_mode]}", True, w_c), (220, 275))

        s.blit(self.bigfont.render(f"{int(self.display_speed)}", True, (255,255,255)), (cx-20, 30))
        s.blit(self.font.render("km/h", True, (200,200,200)), (cx+30, 45))

        if self.log:
            txt = f"Acción: {self.action_text} ({self.log[-1].get('act_val',0):.1f})"
        else: txt = f"Acción: {self.action_text}"
        s.blit(self.font.render(txt, True, (255,200,100)), (SCREEN_W-350, 30))

        bx = SCREEN_W - 100; by = SCREEN_H - 150
        pygame.draw.rect(s, (50,50,50), (bx, by, 20, 100))
        vh = int(self.visibility)
        pygame.draw.rect(s, (0,255,0), (bx, by + (100-vh), 20, vh))
        s.blit(self.font.render("Vis", True, (150,150,150)), (bx, by+110))
        pygame.draw.rect(s, (50,50,50), (bx+40, by, 20, 100))
        dh = int(self.obst_distance_m)
        pygame.draw.rect(s, (255,50,50), (bx+40, by + (100-dh), 20, dh))
        s.blit(self.font.render("Dist", True, (150,150,150)), (bx+40, by+110))
        
        s.blit(self.font.render(f"Grip: {int(self.grip)}%", True, (100,255,100)), (bx-20, by-30))

        inst = self.font.render("SPACE: Demo | S: Guardar | R: Reset | ESC: Salir", True, (180,180,180))
        s.blit(inst, (cx - 200, SCREEN_H - 30))

        pygame.display.flip()

    def save_log_csv(self):
        if not self.log: return
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(self.log).to_csv(f"results/log_{datetime.now().strftime('%H%M%S')}.csv", index=False)
        print("Saved")

    def reset_sim(self):
        self.slider_speed.value = 40; self.slider_dist.value = 40; self.weather_mode = 0; self.display_speed = 40.0

# -------------------- Main --------------------
def main():
    sim = RetroNeonSim()
    sim.run()

if __name__ == "__main__":
    main()