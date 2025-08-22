# """"

# Gestures
# ========
# 👉 Right hand up  → clockwise rotate
# 👈 Left  hand up  → counter‑clockwise rotate
# 🙌 Both hands up  → shield (temporary invulnerability)
# ⬇️ Lean forward   → brake / slow‑mo
# """

import os
import sys
import math
import random
from collections import deque
import cv2
import mediapipe as mp
import pygame
import time

def run_orbit_game():
    # ------------------------- CONFIGURATION ----------------------------
    FULL_WIDTH, HEIGHT = 1000, 700
    PANEL_W, PANEL_H = FULL_WIDTH // 2, HEIGHT
    CENTRE = (PANEL_W + PANEL_W // 2, PANEL_H // 2)
    RADIUS = 200
    PLAYER_SIZE = 15
    ASTEROID_SIZE = 18
    SPAWN_TIME = 1500
    current_rotate_speed = 0.002
    BOOST_MULT = 2.2
    BRAKE_MULT = 0.4
    SHIELD_TIME = 1200
    LEAN_THRESHOLD = 0.08
    USER_WEIGHT = 70.0
    MET_VALUE = 4.0
    CAL_PER_MIN = MET_VALUE * USER_WEIGHT * 3.5 / 200
    HIGH_SCORE_FILE = "highscore.txt"
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_WRIST, R_WRIST = 15, 16
    NOSE = 0

    def load_high_score():
        if os.path.exists(HIGH_SCORE_FILE):
            try:
                with open(HIGH_SCORE_FILE, "r") as f:
                    return int(f.read().strip())
            except ValueError:
                pass
        return 0

    def save_high_score(score: int):
        with open(HIGH_SCORE_FILE, "w") as f:
            f.write(str(score))

    high_score = load_high_score()

    # ----------------------- INITIALIZATION -----------------------------
    pygame.init()
    screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
    pygame.display.set_caption("Orbit Fitness Game – Webcam Gestures 🎮🏃‍♀️")
    fireball_img_orig = pygame.image.load("fireball.png").convert_alpha()
    fireball_img = pygame.transform.scale(fireball_img_orig, (ASTEROID_SIZE * 2, ASTEROID_SIZE * 2))
    player_img_orig = pygame.image.load("ninjastar.png").convert_alpha()
    player_img = pygame.transform.scale(player_img_orig, (PLAYER_SIZE * 5, PLAYER_SIZE * 5))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Calibration
    hip_samples = deque(maxlen=30)
    while len(hip_samples) < 30:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            lh = results.pose_landmarks.landmark[L_HIP].y
            rh = results.pose_landmarks.landmark[R_HIP].y
            hip_samples.append((lh + rh) / 2)
    base_hip_y = sum(hip_samples) / len(hip_samples)

    player_angle = 0.0
    angular_velocity = 0.0
    shield_timer = 0
    spawn_interval = SPAWN_TIME
    asteroids = []
    score_ms = 0
    running = True
    visual_rotation_angle = 0.0
    last_spawn = pygame.time.get_ticks()

    def polar_to_screen(angle: float, radius: float):
        x = CENTRE[0] + radius * math.cos(angle)
        y = CENTRE[1] + radius * math.sin(angle)
        return int(x), int(y)

    def detect_gestures(results):
        g = {"lean": 0, "boost": False, "brake": False, "jump": False}
        if not results.pose_landmarks:
            return g
        lm = results.pose_landmarks.landmark
        nose_y = lm[NOSE].y
        lw_y = lm[L_WRIST].y
        rw_y = lm[R_WRIST].y
        if rw_y < nose_y and lw_y < nose_y:
            g["jump"] = True
        elif rw_y < nose_y:
            g["lean"] = 1
        elif lw_y < nose_y:
            g["lean"] = -1
        hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
        if hip_y > base_hip_y + LEAN_THRESHOLD:
            g["brake"] = True
        return g

    # ------------------------- MAIN LOOP ------------------------------
    while running:
        dt = clock.tick(60)
        visual_rotation_angle = (visual_rotation_angle + 2) % 360
        increase_rate = 0.00000001
        max_speed = 0.005
        current_rotate_speed = min(current_rotate_speed + increase_rate * dt, max_speed)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        gestures = detect_gestures(results)
        angular_velocity += gestures["lean"] * current_rotate_speed
        speed_mult = 1.0
        if gestures["boost"]:
            speed_mult *= BOOST_MULT
        if gestures["brake"]:
            speed_mult *= BRAKE_MULT
        player_angle = (player_angle + angular_velocity * speed_mult * (dt / 16.67)) % (2 * math.pi)
        now = pygame.time.get_ticks()
        if gestures["jump"] and shield_timer <= 0:
            shield_timer = SHIELD_TIME
        if shield_timer > 0:
            shield_timer -= dt
        if now - last_spawn > spawn_interval:
            last_spawn = now
            spawn_interval = max(500, int(spawn_interval * 0.98))
            asteroids.append({"angle": random.uniform(0, 2 * math.pi), "radius": PANEL_W // 2 + 100})
        for a in asteroids:
            a["radius"] -= 1.5 * (dt / 16.67)
        asteroids[:] = [a for a in asteroids if a["radius"] > 0]
        for a in asteroids:
            if abs(a["radius"] - RADIUS) < ASTEROID_SIZE:
                ang_diff = abs((a["angle"] - player_angle + math.pi) % (2 * math.pi) - math.pi)
                if ang_diff < math.radians(12):
                    if shield_timer <= 0:
                        running = False
                    else:
                        a["hit"] = True
        asteroids[:] = [a for a in asteroids if not a.get("hit")]
        score_ms += dt
        screen.fill((0, 0, 0))
        cam_small = cv2.resize(frame, (PANEL_W, PANEL_H))
        cam_rgb = cv2.cvtColor(cam_small, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(cam_rgb.swapaxes(0, 1))
        screen.blit(cam_surface, (0, 0))
        for a in asteroids:
            ax, ay = polar_to_screen(a["angle"], a["radius"])
            screen.blit(fireball_img, fireball_img.get_rect(center=(ax, ay)))
        px, py = polar_to_screen(player_angle, RADIUS)
        angle_degrees = -math.degrees(player_angle) + 90 + visual_rotation_angle
        rotated_img = pygame.transform.rotate(player_img, angle_degrees)
        screen.blit(rotated_img, rotated_img.get_rect(center=(px, py)))
        score_seconds = score_ms // 1000
        calories_burned = CAL_PER_MIN * (score_ms / 60000)
        hud_lines = [
            f"Score: {score_seconds}s",
            f"Best:  {high_score}s",
            f"Calories: {calories_burned:.1f} kcal"
        ]
        for i, text in enumerate(hud_lines):
            surf = font.render(text, True, (240, 240, 240))
            screen.blit(surf, (PANEL_W + 20, 20 + i * 30))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Game over
    final_score_sec = score_ms // 1000
    calories_total = CAL_PER_MIN * (score_ms / 60000)
    if final_score_sec > high_score:
        save_high_score(final_score_sec)
        high_score = final_score_sec  # Update for display
    # -------------------- GAME OVER SCREEN --------------------
    game_over_running = True
    while game_over_running:
      screen.fill((0, 0, 0))
      messages = [
        "GAME OVER",
        f"Score: {final_score_sec}s",
        f"Calories Burned: {calories_total:.1f} kcal",
        f"Highest Score: {high_score}s"
      ]
      for i, msg in enumerate(messages):
        surf = font.render(msg, True, (255, 50, 50) if i == 0 else (240, 240, 240))
        rect = surf.get_rect(center=(FULL_WIDTH//2, 150 + i*60))
        screen.blit(surf, rect)

      pygame.display.flip()
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over_running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_over_running = False
    
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    run_orbit_game()


