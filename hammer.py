"""Hunt-Crossley piano hammer with RK4 integration."""

import numpy as np


class Hammer:
    def __init__(self, mass, K, p, mu, max_vel, sr):
        self.mass = mass
        self.K = K
        self.p = p
        self.mu = mu
        self.max_vel = max_vel
        self.dt = 1.0 / sr
        self.pos = -0.01
        self.vel = 0.0
        self.active = False
        self.in_contact = False
        self.released = False
        self.force = 0.0

    def trigger(self, midi_vel: int):
        v_norm = (midi_vel / 127.0) ** 1.5
        self.vel = self.max_vel * v_norm
        self.pos = -0.01
        self.active = True
        self.in_contact = False
        self.released = False
        self.force = 0.0

    def _contact_force(self, h_pos, h_vel, s_disp):
        comp = h_pos - s_disp
        if comp <= 0:
            return 0.0
        f = self.K * (comp ** self.p) * max(0.0, 1.0 + self.mu * h_vel)
        return min(f, 1e6)

    def tick(self, string_disp: float) -> float:
        if not self.active or self.released:
            self.force = 0.0
            return 0.0

        dt = self.dt
        p, v, sd = self.pos, self.vel, string_disp

        def deriv(pp, vv):
            ff = self._contact_force(pp, vv, sd)
            return vv, -ff / self.mass

        k1v, k1a = deriv(p, v)
        k2v, k2a = deriv(p + .5 * dt * k1v, v + .5 * dt * k1a)
        k3v, k3a = deriv(p + .5 * dt * k2v, v + .5 * dt * k2a)
        k4v, k4a = deriv(p + dt * k3v, v + dt * k3a)

        self.pos = p + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.vel = v + dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)

        self.force = self._contact_force(self.pos, self.vel, sd)

        was = self.in_contact
        self.in_contact = (self.pos - sd) > 0
        if was and not self.in_contact:
            self.released = True
            self.force = 0.0

        return self.force

    @property
    def is_active(self):
        return self.active and not self.released
