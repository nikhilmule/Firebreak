"""Firebreak world with MITRE SimFire's fire physics.

Everything the controllers see and do is unchanged: same map layout, town, drones, water drops,
firebreaks, sectors, scoring. Only `step_fire` is replaced:

* rate of spread from SimFire's Rothermel (1972) implementation, `simfire.world.rothermel`
* real fuel models from `simfire.world.presets` (Anderson 13 fuel models)
* terrain from SimFire's Perlin elevation generator, so slope now speeds fire uphill
* wind as a midflame speed in ft/min, direction in degrees clockwise from north

Each cell burns for SimFire's default fire duration (4 minutes). A cell ignites when the fire travelling into it from a burning neighbour has covered the distance
between cell centres (SimFire's burn-amount rule). Wet cells (drops and firebreaks) cannot ignite,
like SimFire's wetlines. SimFire has no spotting; embers from our model can be kept (default) or
switched off with `embers=False`.

Needs the `simfire` package (Python 3.9): see README, "Testing in SimFire".
"""
from __future__ import annotations

import math
import numpy as np

from simfire.world import presets
from simfire.world.parameters import FuelParticle
from simfire.world.rothermel import compute_rate_of_spread
from simfire.world.elevation_functions import perlin

from .world import World, N, CELL, NONE, GRASS, FOREST, HOUSE, WATER, ROAD, SITE, UNB, BURN, BURNT, WET, \
    BURN_T, BURNABLE, NBRS

FT_PER_M = 3.28084
CELL_FT = CELL * FT_PER_M
# our map classes -> Anderson fuel models in SimFire
FUELS = {
    NONE: presets.NBBarren,
    GRASS: presets.ShortGrass,                   # FM1 short grass
    FOREST: presets.TimberLitterUnderstory,      # FM10 timber with litter and understory
    HOUSE: presets.GrassTimberShrubOverstory,    # FM2: lots ignite through yard grass and shrubs
    WATER: presets.NBWater,
    ROAD: presets.NBUrban,
    SITE: presets.GrassTimberShrubOverstory,
}


class SimFireWorld(World):
    def __init__(self, seed=1, drones=6, fires=18, wind=7.0, params=None, *, embers=True,
                 moisture=0.06, waf=0.4, relief_ft=(0.0, 400.0), burn_s=240):
        self.sf = dict(embers=embers, moisture=moisture, waf=waf, relief_ft=relief_ft, burn_s=burn_s)
        super().__init__(seed, drones, fires, wind, params)

    # ------------------------------------------------------------- layers
    def _make_map(self, wind):
        super()._make_map(wind)
        fp = FuelParticle()
        self._fp = fp
        f = self.fuel
        get = lambda attr: np.vectorize(lambda k: getattr(FUELS[int(k)], attr))(f).astype(np.float64)
        self._w0, self._delta, self._Mx, self._sigma = get("w_0"), get("delta"), get("M_x"), get("sigma")
        lo, hi = self.sf["relief_ft"]
        if hi > lo:
            fn = perlin(octaves=3, persistence=0.7, lacunarity=2.0, seed=int(self.seed % 10000),
                        range_min=lo, range_max=hi)
            elev = np.array([[fn(x, y) for x in range(N)] for y in range(N)], dtype=np.float64)
        else:
            elev = np.zeros((N, N))
        self.elevation = elev
        gy, gx = np.gradient(elev, CELL_FT)          # same slope convention as SimFire
        self._slope_mag = np.sqrt(gx ** 2 + gy ** 2)
        self._slope_dir = np.arctan2(gy, gx + 1e-6)
        self._burn_amt = np.zeros((N, N))

    def _ignite_initial(self):
        super()._ignite_initial()
        self.timer[self.state == BURN] = self.sf['burn_s']

    # --------------------------------------------------------------- fire
    def step_fire(self):
        p, r = self.p, self.rng
        st, fuel = self.state, self.fuel
        if p.wind_shift_deg and self.t == p.wind_shift_t:
            shift = math.radians(p.wind_shift_deg)
            self.wind_dir0 += shift
            self.wind_dir += shift
            self._compute_threat()
        self.wind_dir += (r.random() - 0.5) * 0.02
        self.wind_dir = min(self.wind_dir0 + 0.45, max(self.wind_dir0 - 0.45, self.wind_dir))
        wx, wy = math.cos(self.wind_dir), math.sin(self.wind_dir)
        # grid +y points south; SimFire wants degrees clockwise from north, the way the wind blows
        u_dir = math.degrees(math.atan2(wx, -wy)) % 360
        u = self.wind * 60 * FT_PER_M * self.sf["waf"]           # m/s -> ft/min at midflame

        B = st == BURN
        self.timer[B] -= 1
        ended = B & (self.timer <= 0)
        if ended.any():
            st[ended] = BURNT
            self.burnt_cells += int(ended.sum())
            self.value_lost += float(self.cell_value[ended].sum())
            self.homes_lost += int((ended & (fuel == HOUSE)).sum())
        B = st == BURN

        sy, sx = np.nonzero(B)
        cand = (st == UNB) & BURNABLE[fuel]
        gain = np.zeros((N, N))
        need = np.full((N, N), CELL_FT)
        if len(sy):
            for dx, dy in NBRS:
                ty, tx = sy + dy, sx + dx
                ok = (ty >= 0) & (ty < N) & (tx >= 0) & (tx < N)
                ty, tx, fy, fx = ty[ok], tx[ok], sy[ok], sx[ok]
                ok = cand[ty, tx]
                ty, tx, fy, fx = ty[ok], tx[ok], fy[ok], fx[ok]
                if not len(ty):
                    continue
                k = len(ty)
                R = compute_rate_of_spread(
                    fx.astype(float), fy.astype(float), tx.astype(float), ty.astype(float),
                    self._w0[ty, tx], self._delta[ty, tx], self._Mx[ty, tx], self._sigma[ty, tx],
                    np.full(k, self._fp.h), np.full(k, self._fp.S_T), np.full(k, self._fp.S_e),
                    np.full(k, self._fp.p_p), np.full(k, self.sf["moisture"]), np.full(k, u),
                    np.full(k, u_dir), self._slope_mag[ty, tx], self._slope_dir[ty, tx])
                R = np.nan_to_num(R) / 60.0                              # ft per second
                np.maximum.at(gain, (ty, tx), R)
                if dx and dy:
                    need[ty, tx] = CELL_FT * math.sqrt(2)
        self._burn_amt += gain
        self._burn_amt[~cand] = 0.0
        new = cand & (self._burn_amt >= need)

        if self.sf["embers"]:
            src = B & ((fuel == FOREST) | (fuel == HOUSE))
            ys, xs = np.nonzero(src & (r.random((N, N)) < p.ember_rate * self.wind))
            for y, x in zip(ys, xs):
                dist = 4 + r.random() * 10
                a = self.wind_dir + (r.random() - 0.5) * 0.5
                ex, ey = round(x + math.cos(a) * dist), round(y + math.sin(a) * dist)
                if 0 <= ex < N and 0 <= ey < N and st[ey, ex] == UNB and BURNABLE[fuel[ey, ex]]:
                    new[ey, ex] = True

        st[new] = BURN
        self.timer[new] = self.sf['burn_s']
        self._burn_amt[new] = 0.0

        W = st == WET
        self.timer[W] -= 1
        st[W & (self.timer <= 0)] = UNB
