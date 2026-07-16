"""Piano damper — engages on note-off, ramps up over attack_time."""


class Damper:
    def __init__(self, attack_time, max_damp, sr):
        self.attack_samples = max(1, int(attack_time * sr))
        self.max_damp = max_damp
        self.engaged = False
        self.counter = 0
        self.amount = 0.0

    def note_on(self):
        self.engaged = False
        self.counter = 0
        self.amount = 0.0

    def note_off(self):
        self.engaged = True
        self.counter = 0

    def tick(self, pedal: bool) -> float:
        if pedal or not self.engaged:
            self.amount = 0.0
            return 0.0
        self.counter += 1
        self.amount = min(1.0, self.counter / self.attack_samples)
        return self.amount


class NullDamper:
    def note_on(self): pass
    def note_off(self): pass
    def tick(self, pedal): return 0.0
