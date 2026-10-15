"""Wildfire + drone-fleet simulation (NumPy port of the Firebreak web lab).

Units: metres and seconds. One call to `World.tick()` advances one second.
Grid arrays are indexed [y, x]; flat cell ids are y * N + x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

N = 96            # cells per side
CELL = 40.0       # metres per cell -> 3.84 km map

# fuel types
NONE, GRASS, FOREST, HOUSE, WATER, ROAD, SITE = range(7)
# cell states
UNB, BURN, BURNT, WET = range(4)

BASE_P = np.array([0, 0.013, 0.008, 0.008, 0, 0, 0.008])   # ignition chance / neighbour / s, calm air
BURN_T = np.array([0, 22, 70, 90, 0, 0, 110], dtype=np.int32)  # seconds a cell burns
VALUE = np.array([0, 0, 0, 1, 0, 0, 5], dtype=np.float64)    # home = 1, school / clinic cell = 5
BURNABLE = np.array([False, True, True, True, False, False, True])
NBRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
MODES = ["ready", "assigned", "refill", "rtb", "charge", "lost"]


@dataclass
class Params:
    # drones
    vmax: float = 18.0          # m/s airspeed
    amax: float = 5.0           # m/s^2
    tank: int = 4               # water loads
    endurance: float = 900.0    # s of battery
    drop_r: float = 1.7         # cells doused per drop (radius)
    refill_t: float = 3.0       # s per load at the lake / base
    charge_rate: float = 15.0   # battery seconds restored per second at base
    drop_cooldown: float = 3.0
    wind_coupling: float = 0.6  # share of wind speed that pushes a drone
    # fire
    spread_k: float = 0.18      # wind sensitivity of spread
    ember_rate: float = 0.0005  # ember chance per burning forest/house cell per s per m/s of wind
    wet_time: int = 150         # s a doused cell stays wet
    break_time: int = 420       # s a firebreak drop (pre-wetted ground ahead of a fire) stays wet
    break_dist: int = 4         # cells downwind of the fire's most dangerous edge to lay a break
    commit_breaks: bool = True  # a drone flying a firebreak run finishes it before taking new orders
    # stress conditions (all off by default)
    obs_noise: float = 0.0      # sd of multiplicative error on what the planner sees (fire size, value, success chance)
    miss_rate: float = 0.0      # chance each fire sector is missing from the planner's picture at a decision
    fail_per_hour: float = 0.0  # drone failure rate; a failed drone is down for the rest of the run
    sense_range: float = 0.0    # metres; > 0 means a drone only knows about fires within this distance
    wind_shift_deg: float = 0.0 # sudden wind direction change ...
    wind_shift_t: int = 600     # ... at this second of fire time
    # episode
    tmax: int = 1500
    replan: int = 3             # s between decisions
    bin: int = 8                # fire sector size in cells


@dataclass
class Drone:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    batt: float = 900.0
    water: int = 4
    mode: str = "ready"
    target: int = -1      # sector (bin) id
    aim: int = -1         # cell id
    task: str = "attack"  # "attack" = douse flames, "break" = wet ground ahead of the fire
    drop_cd: float = 0.0
    work: float = 0.0
    drops: int = 0


@dataclass
class Targets:
    """Fire sectors at one decision point. All arrays have length T."""
    bin: np.ndarray
    n: np.ndarray          # burning cells in sector
    value: np.ndarray      # value at risk v_j
    aim: np.ndarray        # most dangerous burning cell (flat id)

    def __len__(self):
        return len(self.bin)


class World:
    def __init__(self, seed: int = 1, drones: int = 6, fires: int = 18, wind: float = 7.0,
                 params: Params | None = None):
        self.p = params or Params()
        self.seed, self.n_drones, self.n_fires = seed, drones, fires
        self.rng = np.random.default_rng(seed)                 # environment randomness
        self.ctl_rng = np.random.default_rng(seed * 104729 + 7)  # controller randomness
        self._make_map(wind)
        self.t = 0
        self.value_lost = 0.0
        self.homes_lost = 0
        self.burnt_cells = 0
        self.extinguished = 0
        self.switches = 0
        self.breaks = 0
        self.events = []          # (t, x, y, task) for every water drop, used by replays
        p_ = self.p
        self.noise_rng = np.random.default_rng(seed * 31337 + 11)   # planner's sensing errors
        fr = np.random.default_rng(seed * 7 + 3)                     # same failure times for every controller
        self.fail_t = (fr.exponential(3600.0 / p_.fail_per_hour, drones) if p_.fail_per_hour > 0
                       else np.full(drones, np.inf))
        self.failures = 0
        self.done = False
        self._ignite_initial()
        bx, by = self.base_xy
        self.drones = [Drone(i, bx + (i - (drones - 1) / 2) * 0.8 * CELL, by,
                             batt=self.p.endurance, water=self.p.tank) for i in range(drones)]
        r = self.p.drop_r
        self._drop_offsets = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)
                              if math.hypot(dx, dy) <= r]

    # ------------------------------------------------------------------ map
    def _make_map(self, wind):
        r = self.rng
        G = 13
        coarse = r.random((G, G))
        g = np.arange(N) / N * (G - 1)
        i0 = np.floor(g).astype(int)
        f = g - i0
        s = f * f * (3 - 2 * f)
        i1 = np.minimum(i0 + 1, G - 1)
        # bilinear with smoothstep: rows = y, cols = x
        top = coarse[i0][:, i0] * (1 - s)[None, :] + coarse[i0][:, i1] * s[None, :]
        bot = coarse[i1][:, i0] * (1 - s)[None, :] + coarse[i1][:, i1] * s[None, :]
        noise = top * (1 - s)[:, None] + bot * s[:, None]
        fuel = np.where(noise > 0.52, FOREST, GRASS).astype(np.int8)

        yy, xx = np.mgrid[0:N, 0:N]
        self.lake = dict(x=14 + r.random() * 10, y=48 + r.random() * 10, rx=6, ry=4)
        L = self.lake
        fuel[((xx - L["x"]) / L["rx"]) ** 2 + ((yy - L["y"]) / L["ry"]) ** 2 <= 1] = WATER

        self.town = dict(x0=28, x1=68, y0=72, y1=86)
        T = self.town
        for y in range(T["y0"], T["y1"] + 1):
            for x in range(T["x0"], T["x1"] + 1):
                street = (x - T["x0"]) % 5 == 0 or (y - T["y0"]) % 5 == 0
                fuel[y, x] = ROAD if street else (HOUSE if r.random() < 0.82 else GRASS)
        for sx, sy in [(36, 77), (57, 82)]:
            fuel[sy:sy + 2, sx:sx + 3] = SITE
        fuel[89:94, 45:52] = ROAD
        self.base = (48, 91)
        self.base_xy = ((48 + 0.5) * CELL, (91 + 0.5) * CELL)
        self.fuel = fuel
        self.state = np.zeros((N, N), np.int8)
        self.timer = np.zeros((N, N), np.int32)

        self.cell_value = VALUE[fuel]
        self.total_value = float(self.cell_value.sum())
        self.homes = int((fuel == HOUSE).sum())

        self.wind_dir0 = math.pi / 2 + (r.random() - 0.5) * 0.6   # roughly toward the town (+y)
        self.wind_dir = self.wind_dir0
        self.wind = float(wind)

        self._compute_threat()

    def _compute_threat(self):
        """Threat map: value a fire in each cell endangers (distance decay, stretched downwind)."""
        yy, xx = np.mgrid[0:N, 0:N]
        vy, vx = np.nonzero(self.cell_value)
        vv = self.cell_value[vy, vx]
        cy, cx = yy.ravel()[:, None], xx.ravel()[:, None]
        dx, dy = vx[None, :] - cx, vy[None, :] - cy
        d = np.maximum(np.hypot(dx, dy), 0.5)
        cos = (dx * math.cos(self.wind_dir0) + dy * math.sin(self.wind_dir0)) / d
        windy = min(1.0, self.wind / 10)
        self.threat = (vv[None, :] * np.exp(-d * (1 - 0.55 * windy * cos) / 14)).sum(1).reshape(N, N)

    def _ignite_initial(self):
        placed, guard = 0, 0
        while placed < self.n_fires and guard < 5000:
            guard += 1
            x = 3 + int(self.rng.random() * (N - 6))
            y = 3 + int(self.rng.random() * 42)
            if BURNABLE[self.fuel[y, x]] and self.state[y, x] == UNB:
                self.state[y, x] = BURN
                self.timer[y, x] = BURN_T[self.fuel[y, x]]
                placed += 1

    # ----------------------------------------------------------------- fire
    @staticmethod
    def _shift(B, dx, dy):
        """out[y, x] = B[y - dy, x - dx] (no wrap-around)."""
        out = np.zeros_like(B)
        ys, yd = (slice(dy, N), slice(0, N - dy)) if dy >= 0 else (slice(0, N + dy), slice(-dy, N))
        xs, xd = (slice(dx, N), slice(0, N - dx)) if dx >= 0 else (slice(0, N + dx), slice(-dx, N))
        out[ys, xs] = B[yd, xd]
        return out

    def step_fire(self):
        p, r = self.p, self.rng
        st, fuel = self.state, self.fuel
        if self.p.wind_shift_deg and self.t == self.p.wind_shift_t:
            shift = math.radians(self.p.wind_shift_deg)
            self.wind_dir0 += shift
            self.wind_dir += shift
            self._compute_threat()          # the planner's risk map is updated with the new wind
        self.wind_dir += (r.random() - 0.5) * 0.02
        self.wind_dir = min(self.wind_dir0 + 0.45, max(self.wind_dir0 - 0.45, self.wind_dir))
        wx, wy = math.cos(self.wind_dir), math.sin(self.wind_dir)

        B = st == BURN
        self.timer[B] -= 1
        ended = B & (self.timer <= 0)
        if ended.any():
            st[ended] = BURNT
            self.burnt_cells += int(ended.sum())
            self.value_lost += float(self.cell_value[ended].sum())
            self.homes_lost += int((ended & (fuel == HOUSE)).sum())
        B = st == BURN

        base = BASE_P[fuel]
        log_no = np.zeros((N, N))
        for dx, dy in NBRS:
            ln = math.hypot(dx, dy)
            wf = math.exp(p.spread_k * self.wind * (dx * wx + dy * wy) / ln) * (0.7 if ln > 1 else 1.0)
            S = self._shift(B, dx, dy)
            log_no += S * np.log1p(-np.minimum(0.999, base * wf))
        cand = (st == UNB) & BURNABLE[fuel]
        new = cand & (r.random((N, N)) < 1 - np.exp(log_no))

        # embers from burning forest and buildings land downwind
        src = B & ((fuel == FOREST) | (fuel == HOUSE))
        ys, xs = np.nonzero(src & (r.random((N, N)) < p.ember_rate * self.wind))
        for y, x in zip(ys, xs):
            dist = 4 + r.random() * 10
            a = self.wind_dir + (r.random() - 0.5) * 0.5
            sx, sy = round(x + math.cos(a) * dist), round(y + math.sin(a) * dist)
            if 0 <= sx < N and 0 <= sy < N and st[sy, sx] == UNB and BURNABLE[fuel[sy, sx]]:
                new[sy, sx] = True

        st[new] = BURN
        self.timer[new] = BURN_T[fuel[new]]

        W = st == WET
        self.timer[W] -= 1
        st[W & (self.timer <= 0)] = UNB

    # -------------------------------------------------------------- targets
    def build_targets(self) -> Targets:
        ys, xs = np.nonzero(self.state == BURN)
        if len(ys) == 0:
            e = np.zeros(0)
            return Targets(e.astype(int), e.astype(int), e, e.astype(int))
        b = self.p.bin
        nb = N // b
        bins = (ys // b) * nb + (xs // b)
        thr = self.threat[ys, xs]
        uniq, inv = np.unique(bins, return_inverse=True)
        n = np.bincount(inv)
        value = np.bincount(inv, weights=thr) + 0.02 * n
        order = np.lexsort((-thr, inv))            # within each bin, most dangerous first
        first = order[np.searchsorted(inv[order], np.arange(len(uniq)))]
        aim = ys[first] * N + xs[first]
        return Targets(uniq, n, value, aim)

    def assignable(self):
        return [d for d in self.drones if d.mode in ("ready", "assigned") and d.water > 0
                and not (self.p.commit_breaks and d.mode == "assigned" and d.task == "break")]

    def p_matrix(self, avail, tg: Targets) -> np.ndarray:
        """Chance drone i puts out sector j before it outgrows the water it carries."""
        if not avail or len(tg) == 0:
            return np.zeros((len(avail), len(tg)))
        p = self.p
        ws = self.wind * p.wind_coupling
        ax = (tg.aim % N + 0.5) * CELL
        ay = (tg.aim // N + 0.5) * CELL
        dxy = np.array([[d.x, d.y] for d in avail])
        dist = np.hypot(ax[None] - dxy[:, :1], ay[None] - dxy[:, 1:])
        tt = dist / max(4.0, p.vmax - ws * 0.5)
        home = np.hypot(ax - self.base_xy[0], ay - self.base_xy[1]) / (p.vmax - ws)
        batt = np.array([d.batt for d in avail])[:, None]
        water = np.array([d.water for d in avail])[:, None]
        grown = tg.n[None] * (1 + 0.004 * tt * (1 + self.wind / 5))
        P = np.minimum(0.95, water * 7 / grown)
        cur = np.array([d.target for d in avail])[:, None] == tg.bin[None]
        P = np.where(cur, np.minimum(0.95, P * 1.15), P)   # slight preference to finish a job
        P[batt < tt + home[None] + 60] = 0.0
        if p.sense_range > 0:
            P[dist > p.sense_range] = 0.0       # the drone does not know about fires beyond its range
        return P

    def break_cell(self, aim: int) -> int:
        """Unburnt fuel cell break_dist..break_dist+6 cells downwind of `aim`, or -1."""
        x0, y0 = aim % N, aim // N
        cx, cy = math.cos(self.wind_dir), math.sin(self.wind_dir)
        for k in range(self.p.break_dist, self.p.break_dist + 7):
            x, y = round(x0 + cx * k), round(y0 + cy * k)
            if 0 <= x < N and 0 <= y < N and self.state[y, x] == UNB and BURNABLE[self.fuel[y, x]]:
                return int(y * N + x)
        return -1

    def apply_assignment(self, avail, tg: Targets, P, a, tasks=None):
        """a[i] = sector index or -1; tasks[i] = 0 attack / 1 firebreak (None = all attack)."""
        for i, d in enumerate(avail):
            j = int(a[i])
            if j >= 0 and P[i, j] > 0:
                if d.target >= 0 and d.target != tg.bin[j]:
                    self.switches += 1
                d.target, d.aim, d.mode, d.task = int(tg.bin[j]), int(tg.aim[j]), "assigned", "attack"
                if tasks is not None and tasks[i] == 1:
                    bc = self.break_cell(int(tg.aim[j]))
                    if bc >= 0:
                        d.aim, d.task = bc, "break"
            elif d.mode == "assigned":
                d.mode, d.target, d.task = "ready", -1, "attack"

    # --------------------------------------------------------------- drones
    def _fly(self, d: Drone, tx, ty, dt=1.0):
        p = self.p
        ws = self.wind * p.wind_coupling
        gx, gy = math.cos(self.wind_dir) * ws, math.sin(self.wind_dir) * ws
        dx, dy = tx - d.x, ty - d.y
        dist = math.hypot(dx, dy)
        vd = min(p.vmax + ws, math.sqrt(2 * p.amax * dist) * 0.9)
        ax = (dx / dist * vd if dist > 0.01 else 0) - gx      # airspeed needed for desired ground velocity
        ay = (dy / dist * vd if dist > 0.01 else 0) - gy
        am = math.hypot(ax, ay)
        if am > p.vmax:
            ax, ay = ax * p.vmax / am, ay * p.vmax / am
        dvx, dvy = ax - d.vx, ay - d.vy
        dm, lim = math.hypot(dvx, dvy), p.amax * dt
        if dm > lim:
            dvx, dvy = dvx * lim / dm, dvy * lim / dm
        d.vx += dvx
        d.vy += dvy
        d.x = min(N * CELL, max(0.0, d.x + (d.vx + gx) * dt))
        d.y = min(N * CELL, max(0.0, d.y + (d.vy + gy) * dt))
        return math.hypot(tx - d.x, ty - d.y)

    def _drop(self, d: Drone, wet_time=None):
        wet_time = wet_time or self.p.wet_time
        self.events.append((self.t, round(d.x, 1), round(d.y, 1), d.task))
        cx, cy = int(d.x // CELL), int(d.y // CELL)
        for dx, dy in self._drop_offsets:
            x, y = cx + dx, cy + dy
            if 0 <= x < N and 0 <= y < N:
                if self.state[y, x] == BURN:
                    self.state[y, x] = WET
                    self.timer[y, x] = wet_time
                    self.extinguished += 1
                elif self.state[y, x] == UNB and BURNABLE[self.fuel[y, x]]:
                    self.state[y, x] = WET
                    self.timer[y, x] = wet_time
                elif self.state[y, x] == WET:
                    self.timer[y, x] = max(self.timer[y, x], wet_time)
        d.water -= 1
        d.drops += 1
        d.drop_cd = self.p.drop_cooldown

    def _drop_line(self, d: Drone):
        """Firebreak drop: a strip across the wind, 9 cells long and 2 deep (about 360 m x 80 m),
        that stays wet for break_time seconds. Burning cells under it are put out too."""
        self.events.append((self.t, round(d.x, 1), round(d.y, 1), "break"))
        cx, cy = d.x / CELL, d.y / CELL
        wx, wy = math.cos(self.wind_dir), math.sin(self.wind_dir)
        px, py = -wy, wx                                   # across the wind
        hit = set()
        for along in range(-4, 5):
            for depth in (0.0, 1.0):
                x = int(cx + px * along + wx * depth)
                y = int(cy + py * along + wy * depth)
                if 0 <= x < N and 0 <= y < N and (x, y) not in hit:
                    hit.add((x, y))
                    st = self.state[y, x]
                    if st == BURN:
                        self.extinguished += 1
                    if st in (BURN, UNB, WET) and BURNABLE[self.fuel[y, x]]:
                        self.state[y, x] = WET
                        self.timer[y, x] = max(self.timer[y, x] if st == WET else 0, self.p.break_time)
        d.water -= 1
        d.drops += 1
        d.drop_cd = self.p.drop_cooldown

    def _refresh_aim(self, d: Drone):
        b, nb = self.p.bin, N // self.p.bin
        by, bx = (d.target // nb) * b, (d.target % nb) * b
        blk = self.state[by:by + b, bx:bx + b] == BURN
        if not blk.any():
            d.aim, d.target, d.mode = -1, -1, "ready"
            return
        thr = np.where(blk, self.threat[by:by + b, bx:bx + b], -1)
        k = int(thr.argmax())
        d.aim = (by + k // b) * N + bx + k % b

    def step_drones(self):
        p = self.p
        bx, by = self.base_xy
        ws = self.wind * p.wind_coupling
        lx, ly = (self.lake["x"] + 0.5) * CELL, (self.lake["y"] + 0.5) * CELL
        for d in self.drones:
            if d.mode != "lost" and self.t >= self.fail_t[d.id]:
                d.mode, d.target = "lost", -1
                self.failures += 1
            if d.mode == "lost":
                continue
            if d.drop_cd > 0:
                d.drop_cd -= 1
            if d.mode != "charge":
                d.batt -= 1 + 0.4 * d.water / p.tank
            if d.batt <= 0:
                d.batt, d.mode = 0.0, "lost"
                continue
            home_t = math.hypot(d.x - bx, d.y - by) / max(4.0, p.vmax - ws) + 45
            if d.mode not in ("charge", "rtb") and d.batt < home_t:
                d.mode, d.target = "rtb", -1
            if d.mode in ("ready", "assigned") and d.water == 0:
                d.mode, d.target = "refill", -1

            if d.mode == "assigned" and d.task == "break":
                if self.state.flat[d.aim] == BURN or self.state.flat[d.aim] == BURNT:
                    d.task = "attack"                     # fire got there first: fight it instead
                else:
                    tx, ty = (d.aim % N + 0.5) * CELL, (d.aim // N + 0.5) * CELL
                    if self._fly(d, tx, ty) < CELL * 1.1 and d.drop_cd <= 0:
                        self._drop_line(d)
                        self.breaks += 1
                        d.mode, d.target, d.task = "ready", -1, "attack"
                    continue
            if d.mode == "assigned":
                self._refresh_aim(d)
                if d.mode == "assigned":
                    tx, ty = (d.aim % N + 0.5) * CELL, (d.aim // N + 0.5) * CELL
                    if self._fly(d, tx, ty) < CELL * 1.1 and d.drop_cd <= 0:
                        self._drop(d)
                else:
                    self._fly(d, d.x, d.y)
            elif d.mode == "refill":
                if self._fly(d, lx, ly) < CELL * 1.5:
                    d.work += 1
                    if d.work >= p.refill_t:
                        d.work = 0
                        d.water += 1
                        if d.water >= p.tank:
                            d.mode = "ready"
            elif d.mode == "rtb":
                if self._fly(d, bx, by) < CELL * 0.6:
                    d.mode = "charge"
            elif d.mode == "charge":
                self._fly(d, bx, by)
                d.batt = min(p.endurance, d.batt + p.charge_rate)
                d.work += 1
                if d.work >= p.refill_t and d.water < p.tank:
                    d.work = 0
                    d.water += 1
                if d.batt >= p.endurance and d.water >= p.tank:
                    d.mode, d.work = "ready", 0
            else:  # ready: hover in place
                self._fly(d, d.x, d.y)

    # ----------------------------------------------------------------- loop
    def tick(self):
        """Advance one second (fire + drones)."""
        if self.done:
            return
        self.step_fire()
        self.step_drones()
        self.t += 1
        if self.t >= self.p.tmax or not (self.state == BURN).any():
            self.done = True

    def decide(self, controller):
        """Run one decision with a controller function (see controllers.py)."""
        tg = self.build_targets()
        avail = self.assignable()
        if avail and len(tg):
            P = self.p_matrix(avail, tg)
            view, Pv, keep = self._planner_view(tg, P)
            if len(view) == 0:
                return
            out = controller(self, avail, view, Pv, self.ctl_rng)
            a, tasks = out if isinstance(out, tuple) else (out, None)
            a = np.array([keep[j] if j >= 0 else -1 for j in a])     # back to the true sector list
            self.apply_assignment(avail, tg, P, a, tasks)

    def _planner_view(self, tg, P):
        """What the planner sees: with stress settings, some sectors are missing and the rest noisy."""
        p, r = self.p, self.noise_rng
        keep = np.arange(len(tg))
        if p.miss_rate > 0:
            keep = keep[r.random(len(tg)) >= p.miss_rate]
        if p.obs_noise <= 0 and len(keep) == len(tg):
            return tg, P, keep
        f = (lambda k: np.exp(r.normal(0, p.obs_noise, k) - p.obs_noise ** 2 / 2)) if p.obs_noise > 0 else (lambda k: np.ones(k))
        view = Targets(tg.bin[keep], np.maximum(1, np.round(tg.n[keep] * f(len(keep)))).astype(int),
                       tg.value[keep] * f(len(keep)), tg.aim[keep])
        Pv = np.minimum(0.95, P[:, keep] * f(len(keep))[None, :])
        return view, Pv, keep

    def step(self, controller):
        if self.t % self.p.replan == 0:
            self.decide(controller)
        self.tick()

    def summary(self) -> dict:
        return dict(
            t=self.t, homes_lost=self.homes_lost, homes=self.homes, value_lost=self.value_lost,
            total_value=self.total_value, value_saved_pct=100 * (1 - self.value_lost / self.total_value),
            area_ha=self.burnt_cells * CELL * CELL / 10000, extinguished=self.extinguished,
            retargets_per_min=self.switches / (self.t / 60) if self.t else 0.0, breaks=self.breaks,
            failures=self.failures,
        )


def run_episode(controller, seed=1, drones=6, fires=18, wind=7.0, params=None) -> dict:
    w = World(seed, drones, fires, wind, params)
    while not w.done:
        w.step(controller)
    return w.summary()
