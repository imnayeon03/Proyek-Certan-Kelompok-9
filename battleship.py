
"""
BATTLESHIP AI - Perang Kapal dengan Kecerdasan Buatan
Mata Kuliah 10S3109 Kecerdasan Buatan - Proyek 1
Kategori algoritma: Uncertainty (inferensi probabilistik / Bayesian)

Cara menjalankan:
    pip install numpy matplotlib pillow
    python battleship_ai.py

Pengaturan ada di bagian "Pilih sumber armada" (FLEET_MODE) dan "Human vs AI" (MODE_INTERAKTIF).
Jendela gambar akan muncul satu per satu; tutup jendela untuk melanjutkan.
"""




import random, csv, os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

N     = 10
COLS  = "ABCDEFGHIJ"
FLEET = {"Carrier": 5, "Battleship": 4, "Cruiser": 3, "Submarine": 3, "Destroyer": 2}
SEED  = 7
os.makedirs("data", exist_ok=True); os.makedirs("assets", exist_ok=True)
print("Siap. Armada:", FLEET)


# ## 1 · Mesin Game (Game Engine)
# Koordinat memakai notasi papan: `E5` = kolom E, baris 5. Internal disimpan sebagai `(baris, kolom)` berindeks 0.



def label(cell):
    r, c = cell
    return f"{COLS[c]}{r + 1}"

def parse_cell(s):
    s = s.strip().upper().replace(" ", "")
    if len(s) < 2 or s[0] not in COLS:
        raise ValueError(f"Koordinat '{s}' tidak valid. Contoh yang benar: E5 (kolom A-J, baris 1-{N}).")
    try:
        r = int(s[1:]) - 1
    except ValueError:
        raise ValueError(f"Koordinat '{s}' tidak valid. Contoh yang benar: E5.")
    if not 0 <= r < N:
        raise ValueError(f"Baris harus antara 1 dan {N}.")
    return (r, COLS.index(s[0]))

def ship_cells(start, length, orient):
    r, c = start
    return [(r, c + i) for i in range(length)] if orient == "H" else [(r + i, c) for i in range(length)]

def cells_ok(cells, occupied):
    return all(0 <= r < N and 0 <= c < N for r, c in cells) and not any(x in occupied for x in cells)


class Board:
    """Satu armada + riwayat tembakan yang diterimanya."""
    def __init__(self, layout):
        self.layout = {k: list(v) for k, v in layout.items()}
        self.owner  = {cell: nm for nm, cells in self.layout.items() for cell in cells}
        self.hits   = {nm: set() for nm in self.layout}
        self.shots  = {}          # (r,c) -> "hit" | "miss"
        self.sunk   = []          # urutan kapal yang tenggelam

    def fire(self, cell):
        if cell in self.shots:
            raise ValueError("Sel itu sudah pernah ditembak.")
        nm = self.owner.get(cell)
        if nm is None:
            self.shots[cell] = "miss"
            return {"result": "miss"}
        self.shots[cell] = "hit"
        self.hits[nm].add(cell)
        if len(self.hits[nm]) == len(self.layout[nm]):
            self.sunk.append(nm)
            return {"result": "sunk", "ship": nm, "cells": list(self.layout[nm])}
        return {"result": "hit"}                 # lawan TIDAK diberi tahu nama kapalnya

    def all_sunk(self):
        return len(self.sunk) == len(self.layout)


# ## 2 · Input Armada: Acak, Manual, dan Dataset
# Aplikasi menerima posisi kapal dengan **tiga cara**:
# 1. **Acak** – `random_layout()`
# 2. **Manual** – teks satu baris per kapal: `NamaKapal TitikAwal Orientasi` (`H` = horizontal, `V` = vertikal), atau lewat prompt `input()` (`ask_manual_layout()`)
# 3. **Dataset** – berkas CSV `data/layouts.csv` berisi banyak susunan armada (`layout_id, ship, start, orient`)



def random_layout(rng=random):
    occupied, layout = set(), {}
    for name, L in FLEET.items():
        while True:
            cells = ship_cells((rng.randrange(N), rng.randrange(N)), L, rng.choice("HV"))
            if cells_ok(cells, occupied):
                layout[name] = cells; occupied.update(cells); break
    return layout

def layout_from_text(text):
    """Baris: 'Carrier A1 H'. Baris kosong dan awalan # diabaikan."""
    layout, occupied = {}, set()
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"Format salah pada '{line}'. Contoh: Carrier A1 H")
        name, start, orient = parts[0].capitalize(), parts[1], parts[2].upper()
        if name not in FLEET:            raise ValueError(f"Kapal '{name}' tidak dikenal. Pilihan: {list(FLEET)}")
        if orient not in ("H", "V"):     raise ValueError("Orientasi harus H atau V.")
        if name in layout:               raise ValueError(f"Kapal {name} ditempatkan dua kali.")
        cells = ship_cells(parse_cell(start), FLEET[name], orient)
        if not cells_ok(cells, occupied):
            raise ValueError(f"{name} keluar papan atau bertabrakan dengan kapal lain.")
        layout[name] = cells; occupied.update(cells)
    missing = [n for n in FLEET if n not in layout]
    if missing:
        raise ValueError(f"Kapal belum ditempatkan: {missing}")
    return layout

def ask_manual_layout():
    """Penempatan interaktif lewat input() di notebook."""
    layout, occupied = {}, set()
    for name, L in FLEET.items():
        while True:
            raw = input(f"{name} (panjang {L}) - titik awal & orientasi, mis. 'A1 H': ")
            try:
                start, orient = raw.split()
                cells = ship_cells(parse_cell(start), L, orient.upper())
                if orient.upper() not in ("H", "V") or not cells_ok(cells, occupied):
                    raise ValueError("Keluar papan / bertabrakan / orientasi salah.")
                layout[name] = cells; occupied.update(cells); break
            except ValueError as e:
                print("  ⚠", e)
    return layout

# ---------- dataset CSV ----------
def _orient(cells):
    return "H" if len(cells) == 1 or cells[0][0] == cells[1][0] else "V"

def save_dataset(path, layouts):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layout_id", "ship", "start", "orient"])
        for lid, lay in enumerate(layouts, 1):
            for name, cells in lay.items():
                w.writerow([lid, name, label(min(cells)), _orient(cells)])

def load_dataset(path):
    groups = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            groups.setdefault(int(row["layout_id"]), []).append(f'{row["ship"]} {row["start"]} {row["orient"]}')
    return {lid: layout_from_text("\n".join(lines)) for lid, lines in groups.items()}

DATASET_PATH = "data/layouts.csv"
if not os.path.exists(DATASET_PATH):                  # buat dataset contoh (100 susunan armada)
    _r = random.Random(2026)
    save_dataset(DATASET_PATH, [random_layout(_r) for _ in range(100)])
DATASET = load_dataset(DATASET_PATH)
print(f"Dataset dimuat: {len(DATASET)} susunan armada dari {DATASET_PATH}")


# ### Pilih sumber armada
# Ubah `FLEET_MODE` menjadi `"random"`, `"manual"`, `"dataset"`, atau `"interactive"`.



FLEET_MODE = "manual"       # "random" | "manual" | "dataset" | "interactive"

MANUAL_LAYOUT_TEXT = """
# NamaKapal  TitikAwal  Orientasi(H/V)
Carrier     A1  H
Battleship  B3  H
Cruiser     D5  V
Submarine   F5  V
Destroyer   H8  H
"""
DATASET_ID = 3              # nomor susunan dari dataset

def pick_layout(mode=None, rng=random):
    mode = mode or FLEET_MODE
    if mode == "random":      return random_layout(rng)
    if mode == "manual":      return layout_from_text(MANUAL_LAYOUT_TEXT)
    if mode == "dataset":     return DATASET[DATASET_ID]
    if mode == "interactive": return ask_manual_layout()
    raise ValueError("FLEET_MODE tidak dikenal")

rng_main = random.Random(SEED)
fleet_A  = pick_layout(FLEET_MODE, rng_main)   # armada pemain / AI-A
fleet_B  = random_layout(rng_main)             # armada lawan (acak)
print(f"Armada A dari mode '{FLEET_MODE}':", {k: (label(v[0]) + "…" + label(v[-1])) for k, v in fleet_A.items()})


# ## 3 · Kecerdasan Buatan (Uncertainty / Bayesian)
# **Ide:** AI tidak tahu di mana kapal berada, tetapi ia tahu *aturan* dan *bukti*. Untuk setiap kapal yang masih
# hidup (panjang $L$), AI menghitung semua penempatan yang **tidak bertentangan** dengan bukti:
# 
# * penempatan tidak boleh menutupi sel **meleset** atau sel milik **kapal yang sudah tenggelam**;
# * penempatan yang menutupi $k$ sel **kena yang belum dijelaskan** diberi bobot $w = B^{k}$ (di sini $B = 30$) — inilah
#   *likelihood* yang membuat AI beralih dari mode **Hunt** (mencari) ke mode **Target** (menghabisi).
# 
# Peta probabilitas sel $(r,c)$:
# 
# $$P(r,c) \;\propto\; \sum_{\text{kapal } s}\ \sum_{\text{penempatan } p \ni (r,c)} B^{\,k(p)}$$
# 
# AI menembak sel belum-ditembak dengan $P$ tertinggi. Tidak ada aturan "kotak-kotak" yang ditulis manual —
# pola *parity* muncul sendiri dari perhitungan probabilitas.



class RandomAI:
    """Baseline: tembak sel acak yang belum pernah ditembak."""
    name = "Random AI"
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()
    def reset(self):
        self.pool = [(r, c) for r in range(N) for c in range(N)]
        self.rng.shuffle(self.pool)
    def choose(self):            return self.pool.pop()
    def update(self, cell, res): pass


class BayesAI:
    """Menembak sel dengan probabilitas kapal tertinggi (inferensi atas semua penempatan konsisten)."""
    name  = "Bayesian AI"
    BOOST = 30.0

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.miss     = np.zeros((N, N), bool)
        self.hit      = np.zeros((N, N), bool)
        self.sunkcell = np.zeros((N, N), bool)
        self.remaining = sorted(FLEET.values(), reverse=True)     # panjang kapal yang masih hidup

    def update(self, cell, res):
        r, c = cell
        if res["result"] == "miss":
            self.miss[r, c] = True
            return
        self.hit[r, c] = True
        if res["result"] == "sunk":
            for rr, cc in res["cells"]:
                self.sunkcell[rr, cc] = True
            self.remaining.remove(len(res["cells"]))

    def probability(self):
        blocked   = self.miss | self.sunkcell
        open_hits = self.hit & ~self.sunkcell                     # kena tapi belum dijelaskan kapal tenggelam
        prob = np.zeros((N, N))
        for L in set(self.remaining):
            mult = self.remaining.count(L)
            for c in range(N - L + 1):                            # penempatan horizontal
                ok = blocked[:, c:c+L].sum(1) == 0
                k  = open_hits[:, c:c+L].sum(1)
                prob[:, c:c+L] += (np.where(ok, self.BOOST ** k, 0.0) * mult)[:, None]
            for r in range(N - L + 1):                            # penempatan vertikal
                ok = blocked[r:r+L, :].sum(0) == 0
                k  = open_hits[r:r+L, :].sum(0)
                prob[r:r+L, :] += (np.where(ok, self.BOOST ** k, 0.0) * mult)[None, :]
        prob[self.miss | self.hit] = 0.0
        return prob

    def choose(self):
        prob = self.probability()
        if prob.max() <= 0:                                       # pengaman
            free = [(r, c) for r in range(N) for c in range(N) if not (self.miss[r, c] or self.hit[r, c])]
            return self.rng.choice(free)
        best = np.argwhere(prob >= prob.max() * (1 - 1e-9))
        r, c = best[self.rng.randrange(len(best))]
        return (int(r), int(c))


def solve(layout, ai):
    """AI memburu satu armada tersembunyi; kembalikan jumlah tembakan sampai semua kapal tenggelam."""
    board = Board(layout); ai.reset(); n = 0
    while not board.all_sunk():
        cell = ai.choose(); res = board.fire(cell); ai.update(cell, res); n += 1
    return n

print("Uji cepat -> Bayes:", solve(fleet_B, BayesAI(random.Random(1))), "tembakan | Random:", solve(fleet_B, RandomAI(random.Random(1))), "tembakan")


# ## 4 · Visualisasi
# Papan laut bergradasi, kapal berlambung runcing dengan menara & anjungan, ledakan bercahaya untuk tembakan **kena**,
# cincin percikan untuk **meleset**, kapal tenggelam berubah merah gelap, dan **heatmap probabilitas** milik AI.



BG, GRID, TXT = "#06101c", "#4a7aa8", "#d6ecff"
HITC = "#ff5a1f"
OCEAN = LinearSegmentedColormap.from_list("ocean", ["#0e4a7b", "#0a3358", "#061f3b"])

def _ocean(ax):
    g = np.linspace(0, 1, 64).reshape(-1, 1)
    ax.imshow(g, cmap=OCEAN, extent=(0, N, N, 0), aspect="auto", zorder=0, interpolation="bilinear")
    x = np.linspace(0, N, 240)
    for i in range(N):
        ax.plot(x, i + .5 + .07 * np.sin(x * 3 + i), color="white", alpha=.05, lw=1, zorder=1)
    ax.hlines(np.arange(N + 1), 0, N, color=GRID, lw=.7, alpha=.55, zorder=2)
    ax.vlines(np.arange(N + 1), 0, N, color=GRID, lw=.7, alpha=.55, zorder=2)

def _style(ax, title):
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal")
    ax.set_xticks(np.arange(N) + .5); ax.set_xticklabels(list(COLS)); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(N) + .5); ax.set_yticklabels(range(1, N + 1))
    ax.tick_params(colors=TXT, length=0, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(1.6)
    ax.set_title(title, color=TXT, fontsize=11.5, fontweight="bold", pad=22)

def _map(cells, pts):
    r0, c0 = min(cells)
    horiz = len(cells) == 1 or cells[0][0] == cells[1][0]
    return [(c0 + u, r0 + v) for u, v in pts] if horiz else [(c0 + v, r0 + u) for u, v in pts]

def _ship(ax, cells, sunk=False):
    L = len(cells)
    fc, ec, dc = ("#3b2430", "#ff5a5a", "#2a1822") if sunk else ("#9aa7b6", "#e6eef7", "#667587")
    hull = [(.12, .2), (L - .5, .2), (L - .08, .5), (L - .5, .8), (.12, .8)]
    ax.add_patch(Polygon(_map(cells, hull), closed=True, fc=fc, ec=ec, lw=1.6, zorder=4, joinstyle="round"))
    u0 = L * .35
    ax.add_patch(Polygon(_map(cells, [(u0, .34), (u0 + .55, .34), (u0 + .55, .66), (u0, .66)]),
                         closed=True, fc=dc, ec=ec, lw=.8, zorder=5))
    for u in ([.55, L - .95] if L >= 3 else [.6]):
        cx, cy = _map(cells, [(u, .5)])[0]
        ax.add_patch(Circle((cx, cy), .1, fc=dc, ec=ec, lw=.8, zorder=5))

def _hit(ax, r, c):
    for rad, a, col in [(.40, .18, HITC), (.29, .35, HITC), (.18, .92, "#ffb703"), (.08, 1, "#fff3c4")]:
        ax.add_patch(Circle((c + .5, r + .5), rad, fc=col, ec="none", alpha=a, zorder=7))

def _miss(ax, r, c):
    ax.add_patch(Circle((c + .5, r + .5), .22, fc="none", ec="#dff3ff", lw=1.3, alpha=.7, zorder=7))
    ax.add_patch(Circle((c + .5, r + .5), .07, fc="#dff3ff", ec="none", alpha=.95, zorder=7))

def _marks(ax, shots):
    for (r, c), s in shots.items():
        (_hit if s == "hit" else _miss)(ax, r, c)

def _status(ax, board):
    h = sum(1 for v in board.shots.values() if v == "hit"); m = len(board.shots) - h
    ax.text(N / 2, N + .75, f"Kena {h}   ·   Meleset {m}   ·   Tenggelam {len(board.sunk)}/{len(FLEET)}",
            ha="center", va="top", color=TXT, fontsize=9.5, clip_on=False)
    for i, nm in enumerate(FLEET):
        alive = nm not in board.sunk
        ax.text(1 + 2 * i, N + 1.7, f"● {nm}", ha="center", va="top", fontsize=7.2, clip_on=False,
                color="#4ade80" if alive else "#f87171")

def draw_board(ax, board, title, reveal="all"):
    """reveal: 'all' = semua kapal terlihat · 'sunk' = hanya kapal tenggelam (kabut perang)."""
    _ocean(ax)
    for nm, cells in board.layout.items():
        sunk = nm in board.sunk
        if reveal == "all" or sunk:
            _ship(ax, cells, sunk)
    _marks(ax, board.shots)
    _style(ax, title); _status(ax, board)

def draw_heat(ax, prob, shots, title):
    _ocean(ax)
    tot = prob.sum()
    mass = prob / tot * 100 if tot > 0 else prob
    ax.imshow(prob / max(prob.max(), 1e-9), cmap="inferno", extent=(0, N, N, 0), alpha=.85,
              zorder=3, vmin=0, vmax=1, interpolation="nearest")
    for r in range(N):
        for c in range(N):
            if (r, c) not in shots and mass[r, c] > 0:
                ax.text(c + .5, r + .5, f"{mass[r, c]:.1f}", ha="center", va="center", fontsize=6.3, zorder=6,
                        color="white" if prob[r, c] / max(prob.max(), 1e-9) < .6 else "black")
    _marks(ax, shots)
    if prob.max() > 0:
        r, c = np.unravel_index(prob.argmax(), prob.shape)
        ax.add_patch(Circle((c + .5, r + .5), .45, fc="none", ec="#38bdf8", lw=2.4, zorder=9))
    _style(ax, title)

def save_fig(fig, name):
    fig.savefig(f"assets/{name}.png", dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")

def show_fleets(a, b, names=("Armada A", "Armada B")):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6), facecolor=BG)
    draw_board(axs[0], Board(a), names[0]); draw_board(axs[1], Board(b), names[1])
    fig.subplots_adjust(wspace=.25, bottom=.14)
    return fig

fig = show_fleets(fleet_A, fleet_B, (f"Armada A  (mode: {FLEET_MODE})", "Armada B  (acak)"))
save_fig(fig, "01_armada"); plt.show()


# ### 4.1 · Bagaimana AI "berpikir": evolusi peta probabilitas
# Angka di tiap sel = persentase probabilitas ada kapal; lingkaran biru = target berikutnya.
# Perhatikan: di awal AI menembak sel tengah (paling banyak kemungkinan penempatan), setelah **kena** peta langsung
# terfokus di sekitarnya (mode *target*), dan setelah kapal **tenggelam** AI kembali ke mode *hunt*.



def hunt_demo(layout, checkpoints=(0, 10, 20, 30), seed=3):
    ai, board = BayesAI(random.Random(seed)), Board(layout)
    snaps, k = [], 0
    if 0 in checkpoints: snaps.append((0, dict(board.shots), ai.probability()))
    while not board.all_sunk():
        cell = ai.choose(); res = board.fire(cell); ai.update(cell, res); k += 1
        if k in checkpoints and not board.all_sunk():
            snaps.append((k, dict(board.shots), ai.probability()))
    fig, axs = plt.subplots(1, len(snaps), figsize=(4.6 * len(snaps), 5.4), facecolor=BG)
    for ax, (k, shots, prob) in zip(np.atleast_1d(axs), snaps):
        draw_heat(ax, prob, shots, f"Setelah {k} tembakan")
    fig.subplots_adjust(wspace=.12)
    return fig

fig = hunt_demo(fleet_B)
save_fig(fig, "02_heatmap_evolusi"); plt.show()


# ## 5 · Pertandingan: Bayesian AI vs Random AI (+ animasi GIF)
# `play_match` memainkan dua armada secara bergantian (1 tembakan per giliran). Seluruh riwayat disimpan
# sehingga bisa diputar ulang menjadi gambar per langkah atau animasi GIF.



def play_match(layout_a, layout_b, shooter_a, shooter_b, names=("Armada A", "Armada B")):
    boards, shooters = [Board(layout_a), Board(layout_b)], [shooter_a, shooter_b]
    for s in shooters: s.reset()
    init_prob = shooter_a.probability() if hasattr(shooter_a, "probability") else None
    history, turn = [], 0
    while True:
        att = turn % 2
        cell = shooters[att].choose()
        res  = boards[1 - att].fire(cell)
        shooters[att].update(cell, res)
        prob = shooters[att].probability() if (att == 0 and hasattr(shooters[0], "probability")) else None
        history.append({"att": att, "cell": cell, "result": res["result"], "ship": res.get("ship"), "prob": prob})
        if boards[1 - att].all_sunk():
            break
        turn += 1
    return {"winner": att, "history": history, "layouts": (layout_a, layout_b), "names": names, "init_prob": init_prob}

def render_match(m, step, figsize=(16.2, 6.6)):
    boards = [Board(l) for l in m["layouts"]]
    h, names = m["history"], m["names"]
    for e in h[:step]:
        boards[1 - e["att"]].fire(e["cell"])
    prob = m["init_prob"]
    for e in h[:step]:
        if e["att"] == 0 and e["prob"] is not None: prob = e["prob"]
    ncol = 3 if prob is not None else 2
    fig = plt.figure(figsize=figsize if ncol == 3 else (11.5, 6.6), facecolor=BG)
    gs = fig.add_gridspec(1, ncol, left=.03, right=.99, top=.78, bottom=.16, wspace=.24)
    draw_board(fig.add_subplot(gs[0]), boards[0], f"{names[0]} — diserang {names[1]}")
    draw_board(fig.add_subplot(gs[1]), boards[1], f"{names[1]} — diserang {names[0]}")
    if ncol == 3:
        draw_heat(fig.add_subplot(gs[2]), prob, boards[1].shots, "Radar probabilitas Bayesian AI")
    fig.text(.5, .955, "BATTLESHIP  ·  AI WAR", ha="center", color="#38bdf8", fontsize=20, fontweight="bold")
    if step == 0:
        sub = "Pertempuran dimulai"
    else:
        e = h[step - 1]; word = {"miss": "MELESET", "hit": "KENA!", "sunk": f"KAPAL {str(e['ship']).upper()} TENGGELAM!"}[e["result"]]
        sub = f"Tembakan #{step}:  {names[e['att']]} → {label(e['cell'])}  ·  {word}"
        if step == len(h): sub += f"   |   PEMENANG: {names[m['winner']]}"
    fig.text(.5, .895, sub, ha="center", color=TXT, fontsize=12.5)
    return fig

def fig_to_pil(fig):
    fig.canvas.draw()
    return PILImage.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

def make_gif(m, path, every=2, size_w=1100):
    frames, steps = [], list(range(0, len(m["history"]), every)) + [len(m["history"])]
    for s in steps:
        fig = render_match(m, s); im = fig_to_pil(fig); plt.close(fig)
        im = im.resize((size_w, int(im.height * size_w / im.width)), PILImage.LANCZOS)
        frames.append(im.convert("P", palette=PILImage.ADAPTIVE, colors=160))
    dur = [260] * (len(frames) - 1) + [3500]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, disposal=2)
    return len(frames)

ai_bayes  = BayesAI(random.Random(11))
ai_random = RandomAI(random.Random(11))
match = play_match(fleet_A, fleet_B, ai_bayes, ai_random, names=("Bayesian AI", "Random AI"))
tot = len(match["history"])
print(f"Pemenang: {match['names'][match['winner']]}  ·  total {tot} tembakan "
      f"(Bayesian: {sum(e['att'] == 0 for e in match['history'])}, Random: {sum(e['att'] == 1 for e in match['history'])})")

for s in (tot // 3, tot):
    fig = render_match(match, s); save_fig(fig, f"03_match_langkah_{s}"); plt.show()

n_frames = make_gif(match, "assets/pertempuran.gif")
print(f"GIF animasi tersimpan: assets/pertempuran.gif ({n_frames} frame)")
print('Buka assets/pertempuran.gif untuk melihat animasi pertempuran.')


# ## 6 · Evaluasi: seberapa pintar AI-nya?
# Kedua AI memburu **100 susunan armada dari dataset** (`data/layouts.csv`). Metrik: jumlah tembakan sampai seluruh
# armada tenggelam (semakin kecil semakin baik; maksimum 100).



t0 = time.time()
ids = sorted(DATASET)
res_bayes  = [solve(DATASET[i], BayesAI(random.Random(i)))  for i in ids]
res_random = [solve(DATASET[i], RandomAI(random.Random(i))) for i in ids]
print(f"Selesai dalam {time.time() - t0:.1f} dtk\n")
print(f"{'AI':<14}{'rata-rata':>10}{'median':>9}{'terbaik':>9}{'terburuk':>10}")
for nm, r in (("Bayesian AI", res_bayes), ("Random AI", res_random)):
    print(f"{nm:<14}{np.mean(r):>10.1f}{np.median(r):>9.0f}{min(r):>9}{max(r):>10}")
print(f"\nBayesian AI lebih hemat ≈ {np.mean(res_random) - np.mean(res_bayes):.1f} tembakan per game "
      f"({(1 - np.mean(res_bayes) / np.mean(res_random)) * 100:.0f}% lebih efisien).")

fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=BG)
ax.set_facecolor("#0b1d33")
bins = np.arange(20, 102, 3)
ax.hist(res_random, bins=bins, color="#f87171", alpha=.85, label=f"Random AI (rata-rata {np.mean(res_random):.1f})")
ax.hist(res_bayes,  bins=bins, color="#38bdf8", alpha=.90, label=f"Bayesian AI (rata-rata {np.mean(res_bayes):.1f})")
ax.axvline(np.mean(res_random), color="#fecaca", ls="--", lw=1.4); ax.axvline(np.mean(res_bayes), color="#bae6fd", ls="--", lw=1.4)
ax.set_xlabel("Jumlah tembakan sampai armada tenggelam", color=TXT); ax.set_ylabel("Jumlah game", color=TXT)
ax.set_title("Distribusi efisiensi: Bayesian AI vs Random AI (100 susunan dataset)", color=TXT, fontweight="bold")
ax.tick_params(colors=TXT)
for s in ax.spines.values(): s.set_color(GRID)
leg = ax.legend(facecolor="#0b1d33", edgecolor=GRID); [t.set_color(TXT) for t in leg.get_texts()]
save_fig(fig, "04_benchmark"); plt.show()


# ## 7 · Human vs AI — main sendiri melawan Bayesian AI
# Ubah `MODE_INTERAKTIF = True`, jalankan sel, lalu ketik koordinat tembakan (mis. `E5`, `j10`). Ketik `q` untuk berhenti.
# Panel kiri = armada Anda (diserang AI), panel kanan = radar lawan (kabut perang: kapal baru terlihat saat tenggelam).
# Jika `False`, sel ini memainkan satu demo otomatis dengan "pemain" yang menembak secara acak.



def play_vs_ai(human_layout, ai_layout, ai=None, scripted=None, render_each=True):
    ai = ai or BayesAI(); ai.reset()
    hb, eb = Board(human_layout), Board(ai_layout)            # hb = armada Anda · eb = armada AI
    word = lambda r: {"miss": "MELESET", "hit": "KENA!", "sunk": "KAPAL TENGGELAM!"}[r["result"]]

    def draw(msg):
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.4), facecolor=BG)
        draw_board(axs[0], hb, "Armada Anda  (diserang AI)")
        draw_board(axs[1], eb, "Radar Lawan  (tembak di sini)", reveal="sunk")
        fig.suptitle("BATTLESHIP  ·  HUMAN vs AI", color="#38bdf8", fontsize=17, fontweight="bold", y=1.02)
        fig.text(.5, .955, msg, ha="center", color=TXT, fontsize=11.5)
        fig.subplots_adjust(wspace=.25, bottom=.15); plt.show()

    msg, winner = "Giliran Anda — tembak!", None
    while winner is None:
        if render_each:
            draw(msg)
        raw = scripted.pop(0) if scripted is not None else input("Tembak koordinat (mis. E5) · 'q' = keluar: ")
        if raw.strip().lower() in ("q", "quit", "keluar"):
            print("Game dihentikan."); return None
        try:
            cell = parse_cell(raw)
            if cell in eb.shots: raise ValueError("Sel itu sudah pernah Anda tembak.")
        except ValueError as e:
            msg = f"⚠ {e}"; continue
        r1 = eb.fire(cell); msg = f"Anda → {label(cell)}: {word(r1)}"
        if eb.all_sunk(): winner = "ANDA"; break
        ac = ai.choose(); r2 = hb.fire(ac); ai.update(ac, r2)
        msg += f"   |   AI → {label(ac)}: {word(r2)}"
        if hb.all_sunk(): winner = "AI"
    draw(msg + f"   |   PEMENANG: {winner}")
    print(f"🏆 Pemenang: {winner}")
    return winner

MODE_INTERAKTIF = False        # <-- ubah ke True untuk bermain sendiri

if MODE_INTERAKTIF:
    play_vs_ai(pick_layout(FLEET_MODE), random_layout())
else:
    demo_rng = random.Random(5)
    spam = [(r, c) for r in range(N) for c in range(N)]; demo_rng.shuffle(spam)
    play_vs_ai(fleet_A, fleet_B, BayesAI(random.Random(2)), scripted=[label(x) for x in spam], render_each=False)


# ## 8 · Ringkasan
# * **Masalah:** menebak posisi kapal tersembunyi dengan informasi parsial → *reasoning under uncertainty*.
# * **Metode:** inferensi probabilistik atas seluruh penempatan yang konsisten dengan bukti; target = sel berprobabilitas tertinggi.
# * **Hasil:** Bayesian AI menenggelamkan armada jauh lebih cepat daripada baseline acak (lihat bagian 6).
# * **Input:** acak, manual (teks/`input()`), dan dataset CSV.
# * **Pengembangan lanjut:** *Monte-Carlo sampling*, lawan yang menempatkan kapal secara adaptif (*game theory*), variasi ukuran papan.