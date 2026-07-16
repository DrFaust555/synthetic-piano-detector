"""
Digital waveguide string — two-rail implementation.

Two numpy arrays (right-going and left-going) of length N.
One extra fractional-delay allpass at the nut for fine tuning.

Each tick:
  1) Read bridge output = right[N-1]
  2) Read nut arrival  = left[N-1]
  3) Apply loss+dispersion at bridge end
  4) Nut reflection = -left[N-1], passed through tuning allpass
  5) Shift both rails (propagate 1 sample)
  6) Insert bridge reflection at right[N-1] end ... NO —
     Actually insert at left[0] (from bridge) and right[0] (from nut).

Tuning:
  Round-trip = 2*N samples.  We want f = sr / (2*N_eff) where
  N_eff = N + allpass_delay + filter_delay.  We choose integer N
  slightly large, then use a first-order allpass to add the
  fractional correction so N + frac = sr/(2*f).
"""

import numpy as np
from dsp import OnePole, DCBlock, Allpass1


class StiffString:
    def __init__(self, freq: float, sr: float, *,
                 detune_cents: float = 0.0,
                 strike_ratio: float = 0.125,
                 loss_g: float = 0.999,
                 lp_coeff: float = 0.3,
                 lp2_coeff: float = 0.1,
                 dispersion_coeffs: list = None):
        self.sr = sr
        f = freq * 2.0 ** (detune_cents / 1200.0)

        # Loss filters (applied once per round-trip at bridge)
        self.lp1 = OnePole(lp_coeff)
        self.lp2 = OnePole(lp2_coeff)
        self.loss_g = loss_g
        self.dc = DCBlock()

        # Dispersion allpass cascade
        self.disp = []
        if dispersion_coeffs:
            for c in dispersion_coeffs:
                if abs(c) > 1e-8:
                    self.disp.append(Allpass1(c))

        # Compute total filter group delay at f (in samples, one-way)
        w = 2 * np.pi * f / sr
        filt_delay = self.lp1.group_delay(f, sr) + self.lp2.group_delay(f, sr)
        for ap in self.disp:
            c = ap.c
            gd = (1 - c * c) / (1 + c * c - 2 * c * np.cos(w))
            filt_delay += gd  # full allpass group delay (≈1 sample each)

        # Desired one-way delay (half round-trip)
        half_rt = sr / (2.0 * f) - filt_delay / 2.0
        half_rt = max(half_rt, 3.0)

        N = int(np.ceil(half_rt))  # integer part (slightly too long)
        frac = N - half_rt           # positive fractional excess

        # Tuning allpass: adds 'frac' samples of delay to shorten the
        # effective period.  We need the allpass to ADD negative delay,
        # i.e., the allpass subtracts 'frac' from the total.
        # Actually: we picked N ≥ half_rt, so the rail is too long by frac.
        # Allpass with coefficient c adds delay = (1-c)/(1+c) at DC ≈ for low f.
        # We want total one-way = half_rt = N - frac.
        # The allpass is in series with the rail, so it ADDS delay.
        # Instead: pick N = floor(half_rt), allpass adds the remainder.
        N = int(np.floor(half_rt))
        frac = half_rt - N  # fractional part to add (0..1)
        if frac < 0.1:
            frac += 1.0
            N -= 1
        N = max(N, 3)

        # Allpass coeff: group delay at DC ≈ (1-c²)/(1-c)² = (1+c)/(1-c).
        # Simpler: for first-order allpass, delay ≈ (1-c)/(1+c) at low freq
        # when c is the coeff in y = c*x + x1 - c*y1.
        # Solving (1-c)/(1+c) = frac → c = (1-frac)/(1+frac)
        self.tuning_ap = Allpass1((1.0 - frac) / (1.0 + frac))

        self.N = N
        self.strike_idx = max(1, min(N - 2, int(N * strike_ratio)))

        # Two rails
        self.R = np.zeros(N, dtype=np.float64)  # right-going: [0]=nut .. [N-1]=bridge
        self.L = np.zeros(N, dtype=np.float64)  # left-going:  [0]=bridge .. [N-1]=nut

        # Output
        # NOTE (naming caveat, not fixed here): `v_bridge` reads as a
        # velocity, but note_engine.py's confirmed-correct hammer injection
        # convention (fn = force/(sr*2*Z), the textbook displacement-wave
        # increment) implies this state is DISPLACEMENT-like, not velocity.
        # If true, downstream consumers of v_bridge that want actual bridge
        # velocity/pressure for radiation (soundboard, output stage) would
        # need an explicit time-derivative, not a raw read. Deliberately not
        # renamed or re-derived here -- out of scope for the disp_scale
        # coupling fix; needs its own investigation before touching
        # soundboard/output semantics. See
        # HAMMER_EXCITATION_ERRATA3_ROOT_CAUSE_CONFIRMED_UA.md.
        self.v_bridge = 0.0

    # -----------------------------------------------------------------
    def tick(self, bridge_refl: float) -> float:
        """
        Advance one sample.

        bridge_refl: reflected wave returning from bridge junction
                     (computed from previous tick's v_bridge).
        Returns: outgoing wave at bridge (v_bridge).
        """
        # 1. Read arriving waves at boundaries
        out_bridge = self.R[self.N - 1]        # arrives at bridge
        out_nut = self.L[self.N - 1]      # arrives at nut

        # 2. Process bridge output through loss + dispersion
        x = out_bridge
        for ap in self.disp:
            x = ap.tick(x)
        x = self.lp1.tick(x)
        x = self.lp2.tick(x)
        x *= self.loss_g
        x = self.dc.tick(x)
        self.v_bridge = x

        # 3. Nut reflection: rigid, invert, through tuning allpass
        nut_refl = self.tuning_ap.tick(-out_nut)

        # 4. Shift rails (propagation)
        self.R[1:] = self.R[:-1]
        self.L[1:] = self.L[:-1]

        # 5. Insert reflected waves
        self.R[0] = nut_refl         # nut → travels rightward to bridge
        self.L[0] = bridge_refl      # bridge → travels leftward to nut

        return self.v_bridge

    # -----------------------------------------------------------------
    def inject_force(self, force: float):
        """Inject hammer force at strike point into both rails."""
        h = force * 0.5
        self.R[self.strike_idx] += h
        # Corresponding point on left rail
        self.L[self.N - 1 - self.strike_idx] += h

    def get_displacement(self) -> float:
        """String displacement at strike point (sum of both rails)."""
        return (self.R[self.strike_idx] +
                self.L[self.N - 1 - self.strike_idx])

    def get_bridge_slope_sq(self) -> float:
        """Squared slope (dy/dx)^2 evaluated near the bridge end.

        Under Kirchhoff-Carrier nonlinearity, the longitudinal force
        injected into the bridge is proportional to this quantity --
        it carries the cross-mode beat content (f_m ± f_n) that produces
        phantom partials.  Computing locally at the bridge (rather than
        spatially averaging) preserves cross-mode coupling: the spatial
        integral kills cross-terms because of mode orthogonality.
        """
        # First-order difference between the two samples closest to the bridge
        dR = self.R[-1] - self.R[-2]
        dL = self.L[-1] - self.L[-2]
        return float(dR * dR + dL * dL)

    def apply_damping(self, amount: float):
        """Apply damper attenuation to entire string."""
        g = 1.0 - amount * 0.015
        self.R *= g
        self.L *= g
