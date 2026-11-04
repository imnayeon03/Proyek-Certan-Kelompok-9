"""
BATTLESHIP PYGAME - Perang Kapal dengan Antarmuka Grafis
Mata Kuliah 10S3109 Kecerdasan Buatan - Proyek 1

Cara menjalankan:
    pip install pygame numpy
    python battleship_pygame.py

Mode yang tersedia:
    1. Human vs AI  - Bermain melawan Bayesian AI
    2. Human vs Human - Dua pemain bergantian di satu komputer

Kontrol:
    - Klik kiri pada grid untuk menembak / menempatkan kapal
    - R = Putar orientasi kapal (saat fase penempatan)
    - ESC = Keluar ke menu
"""

import pygame
import random
import numpy as np
import sys
import os
import csv

# ─────────────────────────── KONSTANTA GAME ───────────────────────────
N     = 10
COLS  = "ABCDEFGHIJ"
FLEET = {"Carrier": 5, "Battleship": 4, "Cruiser": 3, "Submarine": 3, "Destroyer": 2}
SEED  = 7

os.makedirs("data", exist_ok=True)

# ─────────────────────────── WARNA ───────────────────────────
C = {
    "bg":          (6,   16,  28),
    "ocean_dark":  (10,  40,  80),
    "ocean_mid":   (14,  58,  110),
    "ocean_light": (20,  80,  150),
    "grid":        (74,  122, 168),
    "text":        (214, 236, 255),
    "text_dim":    (120, 160, 200),
    "ship":        (154, 167, 182),
    "ship_dark":   (102, 117, 135),
    "ship_sunk":   (59,  36,  48),
    "ship_sunk_e": (255, 90,  90),
    "hit":         (255, 90,  31),
    "hit_glow":    (255, 180, 50),
    "miss":        (200, 230, 255),
    "highlight":   (56,  189, 248),
    "highlight2":  (74,  222, 128),
    "danger":      (248, 113, 113),
    "panel":       (8,   22,  40),
    "panel_border":(50,  90,  140),
    "white":       (255, 255, 255),
    "black":       (0,   0,   0),
    "gold":        (255, 215, 0),
    "preview_ok":  (74,  222, 128, 100),
    "preview_bad": (248, 113, 113, 100),
    "btn":         (20,  60,  110),
    "btn_hover":   (30,  90,  160),
    "btn_active":  (56,  189, 248),
}

# ─────────────────────────── UKURAN LAYAR ───────────────────────────
SW, SH    = 1280, 760
CELL      = 44
GRID_OX   = 60
GRID_OY   = 130
GRID2_OX  = 680
LABEL_SZ  = 18
FPS       = 60


# ══════════════════════════════════════════════════════════════════════
# GAME ENGINE (sama seperti battleship.py)
# ══════════════════════════════════════════════════════════════════════

def label(cell):
    r, c = cell
    return f"{COLS[c]}{r + 1}"

def parse_cell(s):
    s = s.strip().upper().replace(" ", "")
    if len(s) < 2 or s[0] not in COLS:
        raise ValueError(f"Koordinat '{s}' tidak valid.")
    try:
        r = int(s[1:]) - 1
    except ValueError:
        raise ValueError(f"Koordinat '{s}' tidak valid.")
    if not 0 <= r < N:
        raise ValueError(f"Baris harus antara 1 dan {N}.")
    return (r, COLS.index(s[0]))

def ship_cells(start, length, orient):
    r, c = start
    return [(r, c + i) for i in range(length)] if orient == "H" else [(r + i, c) for i in range(length)]

def cells_ok(cells, occupied):
    return all(0 <= r < N and 0 <= c < N for r, c in cells) and not any(x in occupied for x in cells)

def random_layout(rng=random):
    occupied, layout = set(), {}
    for name, L in FLEET.items():
        while True:
            cells = ship_cells((rng.randrange(N), rng.randrange(N)), L, rng.choice("HV"))
            if cells_ok(cells, occupied):
                layout[name] = cells; occupied.update(cells); break
    return layout


class Board:
    def __init__(self, layout):
        self.layout = {k: list(v) for k, v in layout.items()}
        self.owner  = {cell: nm for nm, cells in self.layout.items() for cell in cells}
        self.hits   = {nm: set() for nm in self.layout}
        self.shots  = {}
        self.sunk   = []

    def fire(self, cell):
        if cell in self.shots:
            raise ValueError("Sel sudah pernah ditembak.")
        nm = self.owner.get(cell)
        if nm is None:
            self.shots[cell] = "miss"
            return {"result": "miss"}
        self.shots[cell] = "hit"
        self.hits[nm].add(cell)
        if len(self.hits[nm]) == len(self.layout[nm]):
            self.sunk.append(nm)
            return {"result": "sunk", "ship": nm, "cells": list(self.layout[nm])}
        return {"result": "hit"}

    def all_sunk(self):
        return len(self.sunk) == len(self.layout)


# ══════════════════════════════════════════════════════════════════════
# AI ENGINE
# ══════════════════════════════════════════════════════════════════════

class RandomAI:
    name = "Random AI"
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()
    def reset(self):
        self.pool = [(r, c) for r in range(N) for c in range(N)]
        self.rng.shuffle(self.pool)
    def choose(self): return self.pool.pop()
    def update(self, cell, res): pass
    def probability(self): return np.zeros((N, N))


class BayesAI:
    name  = "Bayesian AI"
    BOOST = 30.0

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.miss      = np.zeros((N, N), bool)
        self.hit       = np.zeros((N, N), bool)
        self.sunkcell  = np.zeros((N, N), bool)
        self.remaining = sorted(FLEET.values(), reverse=True)

    def update(self, cell, res):
        r, c = cell
        if res["result"] == "miss":
            self.miss[r, c] = True; return
        self.hit[r, c] = True
        if res["result"] == "sunk":
            for rr, cc in res["cells"]:
                self.sunkcell[rr, cc] = True
            self.remaining.remove(len(res["cells"]))

    def probability(self):
        blocked   = self.miss | self.sunkcell
        open_hits = self.hit & ~self.sunkcell
        prob = np.zeros((N, N))
        for L in set(self.remaining):
            mult = self.remaining.count(L)
            for c in range(N - L + 1):
                ok = blocked[:, c:c+L].sum(1) == 0
                k  = open_hits[:, c:c+L].sum(1)
                prob[:, c:c+L] += (np.where(ok, self.BOOST ** k, 0.0) * mult)[:, None]
            for r in range(N - L + 1):
                ok = blocked[r:r+L, :].sum(0) == 0
                k  = open_hits[r:r+L, :].sum(0)
                prob[r:r+L, :] += (np.where(ok, self.BOOST ** k, 0.0) * mult)[None, :]
        prob[self.miss | self.hit] = 0.0
        return prob

    def choose(self):
        prob = self.probability()
        if prob.max() <= 0:
            free = [(r, c) for r in range(N) for c in range(N) if not (self.miss[r, c] or self.hit[r, c])]
            return self.rng.choice(free)
        best = np.argwhere(prob >= prob.max() * (1 - 1e-9))
        r, c = best[self.rng.randrange(len(best))]
        return (int(r), int(c))


# ══════════════════════════════════════════════════════════════════════
# PYGAME RENDERER
# ══════════════════════════════════════════════════════════════════════

pygame.init()
pygame.font.init()

def font(size, bold=False):
    return pygame.font.SysFont("Segoe UI", size, bold=bold)

FONT_LG   = font(36, True)
FONT_MD   = font(22, True)
FONT_SM   = font(16)
FONT_XS   = font(13)
FONT_TINY = font(11)

def draw_text(surf, text, x, y, fnt, color, center=False, shadow=True):
    if shadow:
        s = fnt.render(text, True, (0, 0, 0))
        r = s.get_rect()
        if center: r.center = (x, y)
        else:      r.topleft = (x, y)
        surf.blit(s, (r.x + 1, r.y + 1))
    s = fnt.render(text, True, color)
    r = s.get_rect()
    if center: r.center = (x, y)
    else:      r.topleft = (x, y)
    surf.blit(s, r)
    return r

def draw_panel(surf, rect, border=True):
    pygame.draw.rect(surf, C["panel"], rect, border_radius=8)
    if border:
        pygame.draw.rect(surf, C["panel_border"], rect, 2, border_radius=8)

def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i]) * t) for i in range(3))

# ─────────────── Grid Drawing ───────────────

def grid_rect(ox, oy, r, c):
    """Rect piksel untuk sel (r,c) pada grid dengan offset (ox,oy)."""
    return pygame.Rect(ox + c * CELL + LABEL_SZ, oy + r * CELL + LABEL_SZ, CELL, CELL)

def draw_ocean_bg(surf, ox, oy, tick):
    """Latar belakang gradasi laut animasi."""
    for r in range(N):
        for c in range(N):
            t = (r + c) / (2 * N)
            col = lerp_color(C["ocean_dark"], C["ocean_mid"], t)
            # animasi gelombang
            wave = int(8 * (0.5 + 0.5 * np.sin(tick * 0.015 + r * 0.5 + c * 0.3)))
            col = tuple(min(255, col[i] + wave) for i in range(3))
            rect = grid_rect(ox, oy, r, c)
            pygame.draw.rect(surf, col, rect)

def draw_grid(surf, ox, oy):
    """Grid garis + label kolom (A-J) dan baris (1-10)."""
    # label kolom
    for c, ch in enumerate(COLS):
        x = ox + c * CELL + LABEL_SZ + CELL // 2
        draw_text(surf, ch, x, oy + 2, FONT_XS, C["text_dim"], center=True, shadow=False)
    # label baris
    for r in range(N):
        y = oy + r * CELL + LABEL_SZ + CELL // 2
        draw_text(surf, str(r + 1), ox + 4, y, FONT_XS, C["text_dim"], center=False, shadow=False)
    # garis grid
    total = N * CELL
    for i in range(N + 1):
        x = ox + i * CELL + LABEL_SZ
        y = oy + i * CELL + LABEL_SZ
        pygame.draw.line(surf, (*C["grid"], 100), (x, oy + LABEL_SZ), (x, oy + LABEL_SZ + total), 1)
        pygame.draw.line(surf, (*C["grid"], 100), (ox + LABEL_SZ, y), (ox + LABEL_SZ + total, y), 1)
    # border luar
    border_rect = pygame.Rect(ox + LABEL_SZ, oy + LABEL_SZ, total, total)
    pygame.draw.rect(surf, C["grid"], border_rect, 2)

def draw_ship_cell(surf, ox, oy, r, c, sunk=False):
    rect = grid_rect(ox, oy, r, c)
    col_fill = C["ship_sunk"] if sunk else C["ship"]
    col_bord = C["ship_sunk_e"] if sunk else C["ship_dark"]
    pygame.draw.rect(surf, col_fill, rect.inflate(-4, -4), border_radius=3)
    pygame.draw.rect(surf, col_bord, rect.inflate(-4, -4), 2, border_radius=3)
    # highlight
    hi = pygame.Surface((rect.w - 6, 4), pygame.SRCALPHA)
    hi.fill((255, 255, 255, 40))
    surf.blit(hi, (rect.x + 3, rect.y + 3))

def draw_hit_marker(surf, ox, oy, r, c):
    rect = grid_rect(ox, oy, r, c)
    cx, cy = rect.centerx, rect.centery
    for rad, alpha, col in [
        (CELL//2 - 4, 60,  C["hit"]),
        (CELL//2 - 8, 100, C["hit"]),
        (CELL//2 -14, 180, C["hit_glow"]),
        (6,           255, (255, 243, 200)),
    ]:
        s = pygame.Surface((rad*2, rad*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*col, alpha), (rad, rad), rad)
        surf.blit(s, (cx - rad, cy - rad))

def draw_miss_marker(surf, ox, oy, r, c):
    rect = grid_rect(ox, oy, r, c)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.circle(surf, C["miss"], (cx, cy), CELL//2 - 8, 2)
    pygame.draw.circle(surf, C["miss"], (cx, cy), 4)

def draw_board_full(surf, ox, oy, board, reveal_ships, tick, hovered_cell=None, ai_prob=None):
    """
    Render satu papan:
    - reveal_ships: True = tampilkan semua kapal (papan sendiri), False = kabut perang
    - hovered_cell: (r,c) sel yang di-hover mouse
    - ai_prob: array probabilitas AI untuk ditampilkan (opsional)
    """
    draw_ocean_bg(surf, ox, oy, tick)

    # heatmap probabilitas AI
    if ai_prob is not None and ai_prob.max() > 0:
        maxp = ai_prob.max()
        for r in range(N):
            for c in range(N):
                p = ai_prob[r, c] / maxp
                if p > 0.01:
                    s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    # warna inferno: ungu→merah→oranye→kuning
                    if p < 0.33:
                        col = lerp_color((60, 0, 100), (180, 0, 50), p / 0.33)
                    elif p < 0.66:
                        col = lerp_color((180, 0, 50), (240, 100, 0), (p - 0.33) / 0.33)
                    else:
                        col = lerp_color((240, 100, 0), (255, 240, 50), (p - 0.66) / 0.34)
                    s.fill((*col, int(180 * p)))
                    rect = grid_rect(ox, oy, r, c)
                    surf.blit(s, rect.topleft)

    # kapal
    if reveal_ships:
        for nm, cells in board.layout.items():
            sunk = nm in board.sunk
            for (r, c) in cells:
                draw_ship_cell(surf, ox, oy, r, c, sunk)

    # shot markers
    for (r, c), result in board.shots.items():
        if result in ("hit", "sunk"):
            draw_hit_marker(surf, ox, oy, r, c)
        else:
            draw_miss_marker(surf, ox, oy, r, c)

    draw_grid(surf, ox, oy)

    # hover highlight
    if hovered_cell:
        r, c = hovered_cell
        rect = grid_rect(ox, oy, r, c)
        s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        s.fill((*C["highlight"], 60))
        surf.blit(s, rect.topleft)
        pygame.draw.rect(surf, C["highlight"], rect, 2)


def get_hovered_cell(mx, my, ox, oy):
    """Konversi posisi mouse ke (r,c) grid, None jika di luar."""
    gx = mx - ox - LABEL_SZ
    gy = my - oy - LABEL_SZ
    if 0 <= gx < N * CELL and 0 <= gy < N * CELL:
        return (gy // CELL, gx // CELL)
    return None


# ─────────────── Button ───────────────

class Button:
    def __init__(self, rect, text, color=None, hover_color=None, font_obj=None):
        self.rect   = pygame.Rect(rect)
        self.text   = text
        self.color  = color or C["btn"]
        self.hcol   = hover_color or C["btn_hover"]
        self.font   = font_obj or FONT_MD
        self.hovered = False

    def draw(self, surf):
        col = self.hcol if self.hovered else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        pygame.draw.rect(surf, C["highlight"], self.rect, 2, border_radius=8)
        draw_text(surf, self.text, self.rect.centerx, self.rect.centery,
                  self.font, C["text"], center=True)

    def update(self, mx, my):
        self.hovered = self.rect.collidepoint(mx, my)

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered


# ══════════════════════════════════════════════════════════════════════
# LAYAR MENU UTAMA
# ══════════════════════════════════════════════════════════════════════

class MenuScreen:
    def __init__(self, screen):
        self.screen = screen
        cx = SW // 2
        self.btn_pvai = Button((cx - 180, 300, 360, 60), "⚔  Human vs AI", C["btn"], C["btn_hover"], FONT_MD)
        self.btn_pvp  = Button((cx - 180, 390, 360, 60), "👥  Human vs Human", C["btn"], C["btn_hover"], FONT_MD)
        self.btn_quit = Button((cx - 180, 480, 360, 60), "✖  Keluar", (60,15,15), (120,30,30), FONT_MD)
        self.tick = 0

    def handle(self, events):
        mx, my = pygame.mouse.get_pos()
        for btn in [self.btn_pvai, self.btn_pvp, self.btn_quit]:
            btn.update(mx, my)
        for ev in events:
            if ev.type == pygame.QUIT:
                return "quit"
            if self.btn_pvai.clicked(ev):  return "vs_ai"
            if self.btn_pvp.clicked(ev):   return "vs_human"
            if self.btn_quit.clicked(ev):  return "quit"
        return None

    def draw(self):
        self.tick += 1
        surf = self.screen

        # background animasi
        surf.fill(C["bg"])
        for i in range(20):
            y = (self.tick * (i % 5 + 1) * 0.4 + i * 40) % SH
            x = i * 70
            pygame.draw.line(surf, (*C["ocean_dark"], 60), (x, 0), (x + 100, SH), 1)

        # judul
        draw_text(surf, "BATTLESHIP", SW//2, 120, FONT_LG, C["highlight"], center=True)
        draw_text(surf, "PERANG KAPAL — Kecerdasan Buatan", SW//2, 170, FONT_MD, C["text_dim"], center=True)
        draw_text(surf, "Kelompok 9 · Mata Kuliah KA", SW//2, 205, FONT_SM, C["text_dim"], center=True)

        # divider
        pygame.draw.line(surf, C["grid"], (SW//2 - 200, 240), (SW//2 + 200, 240), 2)

        draw_text(surf, "Pilih Mode Permainan", SW//2, 262, FONT_SM, C["text"], center=True)

        for btn in [self.btn_pvai, self.btn_pvp, self.btn_quit]:
            btn.draw(surf)

        # footer
        draw_text(surf, "Bayesian AI · Probabilistic Inference · Uncertainty Reasoning",
                  SW//2, SH - 30, FONT_XS, C["text_dim"], center=True, shadow=False)


# ══════════════════════════════════════════════════════════════════════
# LAYAR PENEMPATAN KAPAL (Ship Placement)
# ══════════════════════════════════════════════════════════════════════

class PlacementScreen:
    """Pemain menempatkan kapalnya satu per satu secara interaktif."""

    def __init__(self, screen, player_name="Pemain"):
        self.screen      = screen
        self.player_name = player_name
        self.layout      = {}
        self.occupied    = set()
        self.ships_list  = list(FLEET.items())   # [(name, length), ...]
        self.current_idx = 0
        self.orient      = "H"
        self.done        = False
        self.tick        = 0
        self.msg         = ""
        self.msg_color   = C["text"]
        self.msg_timer   = 0

        ox = (SW - (N * CELL + LABEL_SZ)) // 2
        self.OX = ox
        self.OY = GRID_OY

        self.btn_random = Button((SW - 220, SH - 70, 200, 45), "🎲 Acak Semua", C["btn"], C["btn_hover"], FONT_SM)
        self.btn_ready  = Button((SW - 440, SH - 70, 200, 45), "✔ Siap!", (20, 70, 30), (30, 110, 50), FONT_SM)

    @property
    def current_ship(self):
        if self.current_idx < len(self.ships_list):
            return self.ships_list[self.current_idx]
        return None

    def set_msg(self, text, color=None, duration=120):
        self.msg = text
        self.msg_color = color or C["danger"]
        self.msg_timer = duration

    def handle(self, events):
        mx, my = pygame.mouse.get_pos()
        self.btn_random.update(mx, my)
        self.btn_ready.update(mx, my)

        hovered = get_hovered_cell(mx, my, self.OX, self.OY)

        for ev in events:
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "menu"
                if ev.key in (pygame.K_r, pygame.K_SPACE):
                    self.orient = "V" if self.orient == "H" else "H"

            if self.btn_random.clicked(ev):
                self._randomize()
                return None

            if self.btn_ready.clicked(ev):
                if len(self.layout) == len(FLEET):
                    return "ready"
                else:
                    self.set_msg("Tempatkan semua kapal dulu!")

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if hovered and self.current_ship:
                    name, L = self.current_ship
                    cells = ship_cells(hovered, L, self.orient)
                    if cells_ok(cells, self.occupied):
                        self.layout[name] = cells
                        self.occupied.update(cells)
                        self.current_idx += 1
                        if self.current_idx >= len(self.ships_list):
                            self.set_msg("Semua kapal ditempatkan! Klik Siap.", C["highlight2"], 300)
                    else:
                        self.set_msg("Tidak bisa di sana — coba posisi lain.")

        return None

    def _randomize(self):
        self.layout   = random_layout()
        self.occupied = set(c for cells in self.layout.values() for c in cells)
        self.current_idx = len(self.ships_list)
        self.set_msg("Armada diacak!", C["highlight2"], 180)

    def draw(self):
        self.tick += 1
        if self.msg_timer > 0:
            self.msg_timer -= 1

        surf = self.screen
        surf.fill(C["bg"])

        # judul
        draw_text(surf, f"PENEMPATAN KAPAL — {self.player_name}",
                  SW // 2, 20, FONT_MD, C["highlight"], center=True)
        draw_text(surf, "Klik grid untuk meletakkan kapal · R = Putar orientasi",
                  SW // 2, 52, FONT_XS, C["text_dim"], center=True, shadow=False)

        # board preview
        mx, my = pygame.mouse.get_pos()
        hovered = get_hovered_cell(mx, my, self.OX, self.OY)

        # ocean
        for r in range(N):
            for c in range(N):
                rect = grid_rect(self.OX, self.OY, r, c)
                col = lerp_color(C["ocean_dark"], C["ocean_mid"], (r+c)/(2*N))
                pygame.draw.rect(surf, col, rect)

        # kapal yang sudah ditempatkan
        for nm, cells in self.layout.items():
            for (r, c) in cells:
                draw_ship_cell(surf, self.OX, self.OY, r, c)

        draw_grid(surf, self.OX, self.OY)

        # preview kapal saat ini
        if hovered and self.current_ship:
            name, L = self.current_ship
            cells = ship_cells(hovered, L, self.orient)
            ok = cells_ok(cells, self.occupied)
            for (r, c) in cells:
                if 0 <= r < N and 0 <= c < N:
                    rect = grid_rect(self.OX, self.OY, r, c)
                    s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    col = C["preview_ok"] if ok else C["preview_bad"]
                    s.fill(col)
                    surf.blit(s, rect.topleft)

        # panel info kapal (kanan)
        px = self.OX + N * CELL + LABEL_SZ + 20
        py = self.OY
        draw_text(surf, "Armada:", px, py, FONT_SM, C["text"])
        for i, (nm, L) in enumerate(self.ships_list):
            placed = nm in self.layout
            current = (i == self.current_idx)
            col = C["highlight2"] if placed else (C["highlight"] if current else C["text_dim"])
            prefix = "✔ " if placed else ("▶ " if current else "  ")
            draw_text(surf, f"{prefix}{nm} ({L})", px, py + 28 + i*26, FONT_XS, col, shadow=False)

        if self.current_ship:
            nm, L = self.current_ship
            draw_text(surf, f"Orientasi: {'Horizontal →' if self.orient=='H' else 'Vertikal ↓'}",
                      px, py + 180, FONT_XS, C["text_dim"], shadow=False)
            draw_text(surf, "(Tekan R untuk putar)", px, py + 200, FONT_TINY, C["text_dim"], shadow=False)

        # pesan
        if self.msg_timer > 0 or self.msg:
            alpha = min(255, self.msg_timer * 4) if self.msg_timer > 0 else 0
            if self.msg_timer <= 0:
                self.msg = ""
            else:
                draw_text(surf, self.msg, SW // 2, SH - 90, FONT_SM, self.msg_color, center=True)

        self.btn_random.draw(surf)
        if len(self.layout) == len(FLEET):
            self.btn_ready.draw(surf)


# ══════════════════════════════════════════════════════════════════════
# LAYAR "TUTUP MATA" (antar giliran Player vs Player)
# ══════════════════════════════════════════════════════════════════════

class HandoffScreen:
    """Layar peralihan saat dua manusia bermain di komputer yang sama."""

    def __init__(self, screen, next_player_name):
        self.screen     = screen
        self.next_name  = next_player_name
        self.btn_cont   = Button((SW//2 - 150, SH//2 + 60, 300, 55),
                                 "▶  Lanjutkan", C["btn"], C["btn_hover"])
        self.tick = 0

    def handle(self, events):
        mx, my = pygame.mouse.get_pos()
        self.btn_cont.update(mx, my)
        for ev in events:
            if ev.type == pygame.QUIT: return "quit"
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                return "continue"
            if self.btn_cont.clicked(ev): return "continue"
        return None

    def draw(self):
        self.tick += 1
        surf = self.screen
        surf.fill((2, 6, 12))
        draw_text(surf, "🔒  GANTI PEMAIN", SW//2, SH//2 - 90, FONT_LG, C["highlight"], center=True)
        draw_text(surf, f"Serahkan layar kepada", SW//2, SH//2 - 20, FONT_MD, C["text_dim"], center=True)
        draw_text(surf, self.next_name, SW//2, SH//2 + 20, FONT_LG, C["text"], center=True)
        draw_text(surf, "(Pemain sebelumnya: jangan lihat!)", SW//2, SH//2 + 55, FONT_XS, C["text_dim"], center=True, shadow=False)
        self.btn_cont.draw(surf)


# ══════════════════════════════════════════════════════════════════════
# LAYAR GAME UTAMA
# ══════════════════════════════════════════════════════════════════════

class GameScreen:
    """
    Layar pertarungan utama.
    mode: "vs_ai" atau "vs_human"
    """

    def __init__(self, screen, mode,
                 layout_p1, layout_p2,
                 name_p1="Pemain 1", name_p2="Pemain 2",
                 ai_difficulty="bayes"):
        self.screen   = screen
        self.mode     = mode
        self.name_p1  = name_p1
        self.name_p2  = name_p2
        self.tick     = 0

        self.board_p1 = Board(layout_p1)   # P1 punya ini; P2 menyerang ini
        self.board_p2 = Board(layout_p2)   # P2 punya ini; P1 menyerang ini

        # AI
        if mode == "vs_ai":
            rng_ai = random.Random()
            self.ai = BayesAI(rng_ai) if ai_difficulty == "bayes" else RandomAI(rng_ai)
        else:
            self.ai = None

        # giliran: 0 = P1 menyerang P2, 1 = P2 menyerang P1
        self.turn     = 0
        self.winner   = None   # None | 0 | 1

        # visual feedback
        self.last_msg        = ""
        self.last_msg_color  = C["text"]
        self.last_msg_timer  = 0
        self.flash_cells     = []   # [(r,c,timer,color)]
        self.show_heatmap    = (mode == "vs_ai")

        # tombol
        self.btn_menu  = Button((SW - 150, 10, 130, 36), "☰  Menu", C["btn"], C["btn_hover"], FONT_XS)
        self.btn_heat  = Button((SW - 300, 10, 130, 36), "🔥 Heatmap", C["btn"], C["btn_hover"], FONT_XS)

        # offset grid
        # Kiri = milik penyerang (reveal all), Kanan = target (fog of war)
        self.OX_LEFT  = GRID_OX
        self.OX_RIGHT = GRID2_OX
        self.OY       = GRID_OY

        # pvp: perlu handoff
        self.pending_handoff = False
        self.handoff_to      = ""

        # animasi AI thinking
        self.ai_thinking      = False
        self.ai_think_timer   = 0

    # ── getter papan sesuai giliran ──
    @property
    def attacker_board(self):
        """Papan milik pemain yang sedang menyerang (papan sendiri, kiri)."""
        return self.board_p1 if self.turn == 0 else self.board_p2

    @property
    def target_board(self):
        """Papan yang diserang (kanan)."""
        return self.board_p2 if self.turn == 0 else self.board_p1

    @property
    def attacker_name(self):
        return self.name_p1 if self.turn == 0 else self.name_p2

    @property
    def target_name(self):
        return self.name_p2 if self.turn == 0 else self.name_p1

    def set_msg(self, text, color=None, dur=200):
        self.last_msg = text
        self.last_msg_color = color or C["text"]
        self.last_msg_timer = dur

    def _fire_at(self, board, cell):
        """Tembak dan kembalikan dict hasil."""
        try:
            return board.fire(cell)
        except ValueError:
            return None

    def _process_shot(self, cell, board):
        """Proses tembakan, tampilkan pesan, cek kemenangan."""
        res = self._fire_at(board, cell)
        if res is None:
            self.set_msg("⚠ Sel sudah ditembak!", C["danger"], 100)
            return False

        r, c = cell
        if res["result"] == "miss":
            self.set_msg(f"💦 {self.attacker_name} → {label(cell)}: MELESET!", C["miss"], 150)
        elif res["result"] == "hit":
            self.set_msg(f"💥 {self.attacker_name} → {label(cell)}: KENA!", C["hit"], 150)
        else:  # sunk
            self.set_msg(f"🔥 {self.attacker_name} → {label(cell)}: KAPAL {res['ship'].upper()} TENGGELAM!", C["hit_glow"], 220)

        if board.all_sunk():
            self.winner = self.turn
        return True

    def handle(self, events):
        self.tick += 1
        mx, my = pygame.mouse.get_pos()
        self.btn_menu.update(mx, my)
        self.btn_heat.update(mx, my)
        if self.last_msg_timer > 0:
            self.last_msg_timer -= 1

        # animasi AI thinking: tunggu timer lalu tembak
        if self.ai_thinking:
            self.ai_think_timer -= 1
            if self.ai_think_timer <= 0:
                self.ai_thinking = False
                # jalankan tembakan AI yang sebenarnya
                self._replay_ai_state()
                cell = self.ai.choose()
                res  = self.board_p1.fire(cell)
                self.ai.update(cell, res)
                r2, c2 = cell
                word = {"miss": "MELESET", "hit": "KENA!", "sunk": f"KAPAL {res.get('ship','').upper()} TENGGELAM!"}[res["result"]]
                self.set_msg(f"🤖 AI → {label(cell)}: {word}", C["hit"] if res["result"] != "miss" else C["miss"], 180)
                if self.board_p1.all_sunk():
                    self.winner = 1
                else:
                    self.turn = 0
            return None

        for ev in events:
            if ev.type == pygame.QUIT: return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return "menu"

            if self.btn_menu.clicked(ev):   return "menu"
            if self.btn_heat.clicked(ev):
                self.show_heatmap = not self.show_heatmap

            if self.winner is not None:
                # klik mana saja saat game over → menu
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    return "menu"
                continue

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # klik di grid target (kanan)
                hovered = get_hovered_cell(mx, my, self.OX_RIGHT, self.OY)
                if hovered:
                    if self.mode == "vs_ai" and self.turn == 0:
                        # giliran manusia
                        ok = self._process_shot(hovered, self.board_p2)
                        if ok and self.winner is None:
                            # jadwalkan giliran AI dengan delay animasi
                            self.turn           = 1
                            self.ai_thinking    = True
                            self.ai_think_timer = 55   # ~0.9 detik pada 60 fps

                    elif self.mode == "vs_human" and not self.pending_handoff:
                        ok = self._process_shot(hovered, self.target_board)
                        if ok and self.winner is None:
                            # Handoff ke pemain berikutnya
                            self.turn = 1 - self.turn
                            self.pending_handoff = True
                            self.handoff_to = self.attacker_name
                            return "handoff"

        return None

    def _replay_ai_state(self):
        """Replay semua tembakan AI agar BayesAI.state konsisten."""
        self.ai.reset()
        for (r, c), result_str in self.board_p1.shots.items():
            fake_res = {"result": result_str}
            if result_str == "sunk":
                nm = self.board_p1.owner.get((r, c))
                if nm:
                    fake_res["cells"] = self.board_p1.layout[nm]
            self.ai.update((r, c), fake_res)


    def draw(self):
        surf = self.screen
        surf.fill(C["bg"])

        # ── Tentukan papan kiri & kanan sesuai giliran ──
        if self.mode == "vs_ai":
            left_board   = self.board_p1   # selalu papan manusia (kiri)
            right_board  = self.board_p2   # selalu papan AI (kanan)
            left_title   = f"⚓ Armada {self.name_p1} (diserang AI)"
            right_title  = f"🎯 Radar: Armada AI (klik untuk tembak)"
            reveal_left  = True
            reveal_right = False
            ai_prob = self.ai.probability() if (self.show_heatmap and hasattr(self.ai, "probability")) else None
        else:
            # pvp: kiri = armada sendiri, kanan = target
            if self.turn == 0:
                left_board   = self.board_p1
                right_board  = self.board_p2
                left_title   = f"⚓ Armada {self.name_p1} (milikmu)"
                right_title  = f"🎯 Radar: Armada {self.name_p2} (tembak di sini!)"
            else:
                left_board   = self.board_p2
                right_board  = self.board_p1
                left_title   = f"⚓ Armada {self.name_p2} (milikmu)"
                right_title  = f"🎯 Radar: Armada {self.name_p1} (tembak di sini!)"
            reveal_left  = True
            reveal_right = False
            ai_prob = None

        mx, my = pygame.mouse.get_pos()
        hovered_right = get_hovered_cell(mx, my, self.OX_RIGHT, self.OY)

        draw_board_full(surf, self.OX_LEFT,  self.OY, left_board,  reveal_left,  self.tick)
        draw_board_full(surf, self.OX_RIGHT, self.OY, right_board, reveal_right, self.tick,
                        hovered_right if self.winner is None else None, ai_prob)

        # judul tiap papan
        draw_text(surf, left_title,  self.OX_LEFT  + (N*CELL+LABEL_SZ)//2, self.OY - 28,
                  FONT_XS, C["text_dim"], center=True, shadow=False)
        draw_text(surf, right_title, self.OX_RIGHT + (N*CELL+LABEL_SZ)//2, self.OY - 28,
                  FONT_XS, C["text_dim"], center=True, shadow=False)

        # header
        draw_text(surf, "BATTLESHIP", SW//2, 10, FONT_MD, C["highlight"], center=True)

        # giliran
        if self.winner is None:
            if self.ai_thinking:
                turn_text = f"🤖 AI sedang berpikir..."
                turn_col  = C["danger"]
            else:
                turn_text = f"Giliran: {self.attacker_name}"
                turn_col  = C["highlight2"] if self.turn == 0 else C["danger"]
            draw_text(surf, turn_text, SW//2, 48, FONT_SM, turn_col, center=True)

        # pesan terakhir
        if self.last_msg_timer > 0:
            draw_text(surf, self.last_msg, SW//2, 80, FONT_SM, self.last_msg_color, center=True)

        # status kapal (bawah tiap papan)
        bot_y = self.OY + N * CELL + LABEL_SZ + 10
        self._draw_ship_status(surf, left_board,  self.OX_LEFT,  bot_y)
        self._draw_ship_status(surf, right_board, self.OX_RIGHT, bot_y)

        # layar menang
        if self.winner is not None:
            self._draw_winner(surf)

        # tombol
        self.btn_menu.draw(surf)
        if self.mode == "vs_ai":
            self.btn_heat.draw(surf)

        # animasi AI thinking overlay
        if self.ai_thinking:
            dots = "." * (1 + (self.tick // 20) % 3)
            draw_text(surf, f"AI berpikir{dots}", SW//2, SH//2, FONT_LG, C["danger"], center=True)

    def _draw_ship_status(self, surf, board, ox, y):
        for i, (nm, cells) in enumerate(FLEET.items()):
            sunk   = nm in board.sunk
            col    = C["danger"] if sunk else C["highlight2"]
            prefix = "✖ " if sunk else "✔ "
            draw_text(surf, f"{prefix}{nm}", ox + i * 90, y, FONT_TINY, col, shadow=False)

    def _draw_winner(self, surf):
        overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))

        win_name = self.name_p1 if self.winner == 0 else self.name_p2
        draw_text(surf, "🏆  GAME OVER!", SW//2, SH//2 - 100, FONT_LG, C["gold"], center=True)
        draw_text(surf, f"PEMENANG: {win_name}", SW//2, SH//2 - 40, FONT_LG, C["highlight"], center=True)

        total_p1 = len(self.board_p2.shots)
        total_p2 = len(self.board_p1.shots)
        hit_p1   = sum(1 for v in self.board_p2.shots.values() if v in ("hit","sunk"))
        hit_p2   = sum(1 for v in self.board_p1.shots.values() if v in ("hit","sunk"))
        draw_text(surf, f"{self.name_p1}: {total_p1} tembakan, {hit_p1} kena",
                  SW//2, SH//2 + 20, FONT_SM, C["text"], center=True)
        draw_text(surf, f"{self.name_p2}: {total_p2} tembakan, {hit_p2} kena",
                  SW//2, SH//2 + 50, FONT_SM, C["text"], center=True)
        draw_text(surf, "[ ESC = Menu ]  [ Klik mana saja = Menu ]",
                  SW//2, SH//2 + 100, FONT_SM, C["text_dim"], center=True, shadow=False)


# ══════════════════════════════════════════════════════════════════════
# LAYAR PILIH KESULITAN AI
# ══════════════════════════════════════════════════════════════════════

class DifficultyScreen:
    def __init__(self, screen):
        self.screen = screen
        cx = SW // 2
        self.btn_bayes  = Button((cx - 180, 300, 360, 60), "🧠  Bayesian AI (Sulit)", C["btn"], C["btn_hover"])
        self.btn_random = Button((cx - 180, 390, 360, 60), "🎲  Random AI (Mudah)",   C["btn"], C["btn_hover"])
        self.btn_back   = Button((cx - 180, 480, 360, 60), "← Kembali",              (40,20,20), (80,30,30))

    def handle(self, events):
        mx, my = pygame.mouse.get_pos()
        for btn in [self.btn_bayes, self.btn_random, self.btn_back]:
            btn.update(mx, my)
        for ev in events:
            if ev.type == pygame.QUIT: return "quit"
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE: return "menu"
            if self.btn_bayes.clicked(ev):  return "bayes"
            if self.btn_random.clicked(ev): return "random"
            if self.btn_back.clicked(ev):   return "menu"
        return None

    def draw(self):
        surf = self.screen
        surf.fill(C["bg"])
        draw_text(surf, "PILIH TINGKAT KESULITAN AI", SW//2, 150, FONT_LG, C["highlight"], center=True)
        draw_text(surf, "Bayesian AI menggunakan inferensi probabilistik untuk menebak posisi kapalmu.",
                  SW//2, 220, FONT_SM, C["text_dim"], center=True, shadow=False)
        draw_text(surf, "Random AI menembak secara acak.",
                  SW//2, 248, FONT_SM, C["text_dim"], center=True, shadow=False)
        for btn in [self.btn_bayes, self.btn_random, self.btn_back]:
            btn.draw(surf)


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def main():
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("BATTLESHIP — Perang Kapal (Kelompok 9)")
    clock  = pygame.time.Clock()

    # State
    state     = "menu"
    menu      = MenuScreen(screen)
    diff_scr  = None
    place_p1  = None
    place_p2  = None
    handoff   = None
    game      = None
    game_mode = None   # "vs_ai" | "vs_human"
    ai_diff   = "bayes"
    layout_p1 = None
    layout_p2 = None

    while True:
        events = pygame.event.get()
        for ev in events:
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        # ── STATE MACHINE ──────────────────────────────────────────

        if state == "menu":
            result = menu.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "vs_ai":
                game_mode = "vs_ai"
                diff_scr  = DifficultyScreen(screen)
                state     = "difficulty"
            elif result == "vs_human":
                game_mode = "vs_human"
                place_p1  = PlacementScreen(screen, "Pemain 1")
                state     = "place_p1"
            menu.draw()

        elif state == "difficulty":
            result = diff_scr.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "menu":
                state = "menu"
            elif result in ("bayes", "random"):
                ai_diff  = result
                place_p1 = PlacementScreen(screen, "Pemain")
                state    = "place_p1"
            diff_scr.draw()

        elif state == "place_p1":
            result = place_p1.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "menu":
                state = "menu"
            elif result == "ready":
                layout_p1 = place_p1.layout
                if game_mode == "vs_ai":
                    layout_p2 = random_layout()
                    game = GameScreen(screen, "vs_ai",
                                      layout_p1, layout_p2,
                                      "Pemain", "AI",
                                      ai_difficulty=ai_diff)
                    state = "game"
                else:
                    handoff   = HandoffScreen(screen, "Pemain 2")
                    place_p2  = PlacementScreen(screen, "Pemain 2")
                    state     = "handoff_p2"
            place_p1.draw()

        elif state == "handoff_p2":
            result = handoff.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "continue":
                state = "place_p2"
            handoff.draw()

        elif state == "place_p2":
            result = place_p2.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "menu":
                state = "menu"
            elif result == "ready":
                layout_p2 = place_p2.layout
                game = GameScreen(screen, "vs_human",
                                  layout_p1, layout_p2,
                                  "Pemain 1", "Pemain 2")
                # handoff ke P1 untuk memulai
                handoff = HandoffScreen(screen, "Pemain 1 (giliran pertama!)")
                state   = "handoff_start"
            place_p2.draw()

        elif state == "handoff_start":
            result = handoff.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "continue":
                state = "game"
            handoff.draw()

        elif state == "game":
            result = game.handle(events)

            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "menu":
                state = "menu"
                menu  = MenuScreen(screen)
            elif result == "handoff":
                # pvp: ganti pemain
                handoff = HandoffScreen(screen, game.attacker_name)
                state   = "game_handoff"

            # klik saat game over → kembali menu
            if game.winner is not None:
                for ev in events:
                    if ev.type == pygame.MOUSEBUTTONDOWN or \
                       (ev.type == pygame.KEYDOWN and ev.key != pygame.K_ESCAPE):
                        state = "menu"
                        menu  = MenuScreen(screen)
            game.draw()

        elif state == "game_handoff":
            result = handoff.handle(events)
            if result == "quit":
                pygame.quit(); sys.exit()
            elif result == "continue":
                state = "game"
            handoff.draw()

        # ──────────────────────────────────────────────────────────
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
