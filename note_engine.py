"""
Single piano note engine.

Assembles hammer, 1-3 unison strings, bridge scattering junction,
soundboard resonator, and damper.

Tick order (critical for correctness):
  1. Propagate strings with stored bridge reflections
  2. Read v_bridge from each string → compute junction
  3. Store new reflections for next tick
  4. Read displacement at strike → compute hammer force
  5. Inject force into strings (AFTER shift, so it's not overwritten)
  6. Damper
  7. Mix output
"""

import os
import warnings
import numpy as np
from string_model import StiffString
from hammer import Hammer
from soundboard import Soundboard
from soundboard_data import SoundboardData
from soundboard_fir import SoundboardFIR
from soundboard_eq import SoundboardEQ
from damper import Damper, NullDamper
from dsp import DCBlock, Biquad
import params


class NoteEngine:
    def __init__(self, midi: int, sr: float = 44100.0):
        self.sr = sr
        self.midi = midi
        self.f0 = params.midi_to_freq(midi)

        n_str = params.num_strings(midi)
        g, lp1, lp2 = params.loss_params(midi, sr)
        disp = params.dispersion_coeffs(midi, sr)
        hp = params.hammer_params(midi)

        self.strings = []
        for i in range(n_str):
            dt = params.detune_cents(midi, i, n_str)
            self.strings.append(StiffString(
                self.f0, sr,
                detune_cents=dt,
                strike_ratio=hp['strike_ratio'],
                loss_g=g, lp_coeff=lp1, lp2_coeff=lp2,
                dispersion_coeffs=disp,
            ))
        self.n_str = n_str

        self.hammer = Hammer(hp['mass'], hp['K'], hp['p'],
                             hp['mu'], hp['max_vel'], sr)

        adm = params.bridge_admittance(midi)
        self.imp_ratio = 1.0 / max(adm, 1e-6)

        from calib.store import get_calibration
        _cal = get_calibration()
        # Soundboard backend selector:
        #   board_kind = 0  -> legacy biquad bank from soundboard_modes()
        #   board_kind = 1  -> modal data-driven (40 modes + per-note coupling)
        #   board_kind = 2  -> FIR convolution with measured average IR
        kind = int(round(_cal.get_global('board_kind')))
        json_ok = os.path.exists(
            os.path.join(os.path.dirname(__file__), 'calib', 'soundboard_ir.json'))
        if kind == 3 and json_ok:
            self.board = SoundboardEQ(midi, sr)
        elif kind == 2 and json_ok:
            self.board = SoundboardFIR(midi, sr)
        elif kind == 1 and json_ok:
            self.board = SoundboardData(midi, sr)
        else:
            self.board = Soundboard(params.soundboard_modes(), sr)
        # Cap direct_mix.  Raw bridge velocity above ~1.0× has a raspy/wheezy
        # character (string-only sound is wire-like); the soundboard should
        # dominate.  Calibrator can still tune below the cap.
        _raw_direct_mix = float(_cal.get_global('direct_mix'))
        self._direct_mix = min(1.0, _raw_direct_mix)
        if _raw_direct_mix > 1.0:
            warnings.warn(
                f'direct_mix clamped at MIDI {midi}: requested '
                f'{_raw_direct_mix:.4g}, runtime cap is 1.0', stacklevel=2)
        self._board_mix = _cal.get_global('board_mix')

        # Longitudinal mode resonator BANK + post-LPF.
        # Multi-modal: comb at k·f_long_1 for k=1..6 with 1/k^decay gain weighting
        # so lower modes (warmer) dominate over upper modes (brighter).
        # Plus a one-pole LPF on the summed phantom signal whose cutoff is a
        # tunable global, so we can dial back high-end longitudinal brightness.
        l_gain, l_freq1, l_Q = params.longitudinal_params(midi, sr)
        self._long_gain = float(l_gain)
        self._long_filters = []
        self._long_mode_weights = []
        decay = max(0.1, _cal.get_global('long_mode_decay'))
        n_long_modes = 6
        for k in range(1, n_long_modes + 1):
            f_k = l_freq1 * k
            if 50.0 < f_k < sr * 0.45:
                self._long_filters.append(Biquad.bandpass(f_k, l_Q, sr))
                self._long_mode_weights.append(1.0 / (k ** decay))
        # Normalise weights so total is ~1.0 (otherwise gain swings with decay)
        wsum = sum(self._long_mode_weights) + 1e-12
        self._long_mode_weights = [w / wsum for w in self._long_mode_weights]
        # DC-block state on the source (slope^2)
        self._long_hp_x1 = 0.0
        self._long_hp_y1 = 0.0
        self._long_hp_R = 0.999
        # Post-LPF on the summed phantom signal
        cutoff = max(200.0, min(sr * 0.45, _cal.get_global('long_lp_cutoff')))
        # one-pole LPF coefficient: y = (1-a)*x + a*y_prev, a = exp(-2*pi*fc/sr)
        self._long_lp_a = float(np.exp(-2 * np.pi * cutoff / sr))
        self._long_lp_y = 0.0
        # Envelope follower for jv — gates phantom output so resonators only
        # contribute when the main bridge signal is active (prevents metallic
        # flare-ups when high-Q modes are re-excited during quiet sustain).
        self._env_jv = 0.0
        self._env_peak = 1e-12
        env_tc = 0.020  # 20 ms attack/release
        self._env_a = float(np.exp(-1.0 / (env_tc * sr)))
        # Slow peak-decay for normalising envelope to [0, 1]
        peak_tc = 1.5
        self._env_peak_a = float(np.exp(-1.0 / (peak_tc * sr)))

        # Final-output spectral tilt: one-pole LPF to roll off HF + low-shelf
        # boost (DC blocker style high-pass complement) to compensate spectral
        # imbalance.  Baseline: 8 kHz LPF, out_hf_keep=1.0 (transparent — the
        # cascade is a no-op).  Do NOT calibrate this jointly with hammer/
        # string-loss/bridge/board parameters: it silently absorbs and masks
        # their fit errors (see PHYSICAL_PIANO_MODELS_AUDIT_UA.md item #1).
        # Only reintroduce a tilt (out_hf_cutoff 6000..9000, out_hf_keep
        # 0.5..1.0) as a deliberate, separately-justified last stage.
        out_lp_fc = max(500.0, min(sr * 0.45, _cal.get_global('out_hf_cutoff')))
        self._out_lp_a = float(np.exp(-2 * np.pi * out_lp_fc / sr))
        self._out_lp_y = 0.0
        # LF boost: scale the LPF output (low-freq content) by `out_lf_boost`.
        # HF keep: scale the HF residue (above cutoff) by `out_hf_keep`.
        self._out_lf_boost = float(_cal.get_global('out_lf_boost'))
        self._out_hf_keep = float(np.clip(_cal.get_global('out_hf_keep'), 0.0, 1.0))

        # Artificial low-frequency resonators (sub-bass body weight).
        lf_freq = max(20.0, min(sr * 0.45, _cal.get_global('lf_res_freq')))
        lf_Q = max(1.0, min(40.0, _cal.get_global('lf_res_Q')))
        self._lf_res_filter = Biquad.bandpass(lf_freq, lf_Q, sr)
        self._lf_res_gain = float(_cal.get_global('lf_res_gain'))
        # Second LF resonator — sub-bass "barrel"
        lf2_freq = max(20.0, min(sr * 0.45, _cal.get_global('lf_res2_freq')))
        lf2_Q = max(1.0, min(40.0, _cal.get_global('lf_res2_Q')))
        self._lf_res2_filter = Biquad.bandpass(lf2_freq, lf2_Q, sr)
        self._lf_res2_gain = float(_cal.get_global('lf_res2_gain'))

        dp = params.damper_params(midi)
        self.damper = (Damper(dp['attack_time'], dp['max_damp'], sr)
                       if dp else NullDamper())
        # Per-string damper-speed multipliers — outer strings dampen faster.
        if n_str == 3:
            self._damper_speeds = [1.0, 1.6, 2.4]
        elif n_str == 2:
            self._damper_speeds = [1.0, 1.8]
        else:
            self._damper_speeds = [1.0]

        self.dc_out = DCBlock()
        self.active = False
        self.pedal = False
        self._refl = [0.0] * n_str
        self._silent = 0
        self._peak_track = 1e-12  # running peak for relative silence threshold

        # String impedance for force normalisation: physical value from
        # Chabassier string table (Z = sqrt(T0 * mu)).
        self.Z = params.string_impedance(midi)
        # Per-string output amplitude multipliers (trichord notes).
        if n_str == 3:
            self._voice_amps = [params.voice_amplitude(i) for i in range(n_str)]
        else:
            self._voice_amps = None

        # Displacement scaling: the waveguide rails (R/L) already store
        # physical displacement, not a velocity-like quantity -- the force
        # injection two lines below this comment used to live,
        # `fn = force / n_str / (sr * 2 * Z)`, is exactly the textbook
        # displacement-wave injection increment dy = F*dt/(2*Z) = F/(2*Z*fs).
        # A prior version of this comment claimed the opposite ("velocity-
        # like, very small") and multiplied by 2*Z*sr here to "convert to
        # physical meters" -- that undid the injection convention and
        # inflated the hammer<->string coupling gain by ~5 orders of
        # magnitude, causing non-physical energy gain during contact
        # (confirmed via a coupling-scale sweep + open-loop ablation; see
        # HAMMER_EXCITATION_ERRATA3_ROOT_CAUSE_CONFIRMED_UA.md). Fixed to 1.0
        # (no rescaling needed) -- locked in by
        # tests/test_hammer_coupling_passivity.py.
        self.disp_scale = 1.0

    # ----------------------------------------------------------------
    def note_on(self, vel: int, pedal: bool = False):
        self.active = True
        self.pedal = pedal
        self._silent = 0
        self._refl = [0.0] * self.n_str
        self.hammer.trigger(vel)
        self.damper.note_on()

    def note_off(self):
        self.damper.note_off()

    def set_pedal(self, down: bool):
        self.pedal = down

    # ----------------------------------------------------------------
    def tick(self) -> float:
        if not self.active:
            return 0.0

        # 1. Propagate strings
        for i, s in enumerate(self.strings):
            s.tick(self._refl[i])

        # 2. Bridge scattering junction with per-string amplitude weighting.
        if self._voice_amps is not None:
            inc = [s.v_bridge * self._voice_amps[i]
                   for i, s in enumerate(self.strings)]
        else:
            inc = [s.v_bridge for s in self.strings]
        s_inc = sum(inc)
        jv = 2.0 * s_inc / (self.n_str + self.imp_ratio)

        # 3. Store reflections
        for i in range(self.n_str):
            self._refl[i] = jv - inc[i]

        # 4. Hammer — scale displacement to physical units (meters)
        disp_wg = sum(s.get_displacement() for s in self.strings) / self.n_str
        disp_phys = disp_wg * self.disp_scale
        force = self.hammer.tick(disp_phys)

        # 5. Inject force AFTER propagation shift
        if force > 0:
            fn = force / self.n_str / (self.sr * 2.0 * self.Z)
            for s in self.strings:
                s.inject_force(fn)

        # 6. Damper — per-string staggered damping speeds.  Outer unison
        # strings (idx 2, then 1) get damped faster than the main central
        # string (idx 0), reproducing the audible "tail" of a real piano
        # release where the side strings die first and the main string
        # rings on.  Speeds for n=3: [1.0, 1.6, 2.4] (main, mid, outer).
        damp = self.damper.tick(self.pedal)
        if damp > 0:
            for i, s in enumerate(self.strings):
                speed = self._damper_speeds[i] if i < len(self._damper_speeds) else 1.0
                s.apply_damping(min(1.0, damp * speed))

        # 7. Output: weighted mix of direct bridge velocity + soundboard radiation.
        # Real piano sound is dominated by the soundboard; raw string velocity
        # alone is wire-like.  The mix weights are calibrated globals.
        board = self.board.tick(jv)
        out = jv * self._direct_mix + board * self._board_mix

        # Artificial low-frequency body resonators (two parallel modes)
        if self._lf_res_gain > 0.0:
            out += self._lf_res_filter.tick(jv) * self._lf_res_gain
        if self._lf_res2_gain > 0.0:
            out += self._lf_res2_filter.tick(jv) * self._lf_res2_gain

        # 7a. Longitudinal phantom partials (Kirchhoff-Carrier nonlinearity).
        # Per Chaigne/Chabassier: the longitudinal force at the bridge is
        # proportional to (dy/dx)^2 evaluated near the bridge end.  This local
        # quantity carries cross-mode beat content (f_m ± f_n) which, on an
        # inharmonic string, generates the "phantom partials" between integer
        # harmonics — the colouration that distinguishes piano tone from a
        # plain stiff-string model.
        if self._long_gain > 0.0 and self._long_filters:
            slope_sq = sum(s.get_bridge_slope_sq() for s in self.strings)
            src = slope_sq * 1.0e8
            # DC-block (1st-order HP) on the source
            self._long_hp_y1 = (
                src - self._long_hp_x1 + self._long_hp_R * self._long_hp_y1
            )
            self._long_hp_x1 = src
            # Weighted sum across longitudinal modes (lower-k louder)
            phantom = 0.0
            for filt, w in zip(self._long_filters, self._long_mode_weights):
                phantom += filt.tick(self._long_hp_y1) * w
            # Post-LPF to roll off high-frequency longitudinal brightness
            self._long_lp_y = (1.0 - self._long_lp_a) * phantom + \
                              self._long_lp_a * self._long_lp_y
            # ----- Envelope gate -----
            # Track running |jv| envelope and its slow-decay peak.  Gate the
            # phantom by env/peak so resonators only contribute proportional
            # to the *current* loudness — no flare-ups in quiet sustain.
            a_jv = abs(jv)
            self._env_jv = self._env_a * self._env_jv + (1.0 - self._env_a) * a_jv
            if self._env_jv > self._env_peak:
                self._env_peak = self._env_jv
            else:
                self._env_peak *= self._env_peak_a
            gate = self._env_jv / (self._env_peak + 1e-15)
            # Soft-shape the gate so small jv attenuates phantom strongly
            gate = gate * gate
            out += self._long_lp_y * self._long_gain * gate

        # NOTE: the final output spectral tilt (LF-shelf boost / HF-shelf cut)
        # and DC blocking are applied in render(), AFTER the FIR/EQ soundboard
        # convolution is mixed in — not here.  Previously this stage ran here,
        # per-sample, on `out` alone; the FIR contribution was added later in
        # render() and therefore never passed through the tilt or DC block at
        # all (see audit item #2: direct got the tilt, FIR didn't).  tick()
        # now returns the untilted model signal; _apply_output_stage() in
        # render() gives direct+board(IIR)+FIR identical final processing.

        # Auto-deactivate when silent — threshold RELATIVE to running peak so we
        # don't kill the tail when the absolute level is small (which it is for
        # this model: peaks ~1e-5 even at v127).  We track a slow-decaying peak
        # and stop only once we're 70 dB below it AND the hammer is released.
        a = abs(out)
        if a > self._peak_track:
            self._peak_track = a
        else:
            self._peak_track *= 0.999998  # ~30s decay
        thr = self._peak_track * 3.16e-4  # -70 dB rel. peak
        if a < thr and not self.hammer.is_active:
            self._silent += 1
            if self._silent > self.sr * 0.5:
                self.active = False
        else:
            self._silent = 0

        return out

    # ----------------------------------------------------------------
    def _apply_output_stage(self, x: np.ndarray) -> np.ndarray:
        """Final-stage spectral tilt + DC block, vectorised.

        Applied ONCE, in render(), to the fully combined model signal
        (direct bridge + board IIR + FIR/EQ convolution) so every path gets
        identical output processing — see the note in tick() and audit
        item #2 (previously the FIR path bypassed this stage entirely).
        Mathematically equivalent to the per-sample recursions this used to
        run inside tick() before the FIR contribution existed.
        """
        from scipy.signal import lfilter
        a = self._out_lp_a
        lp = lfilter([1.0 - a], [1.0, -a], x)
        tilted = self._out_lf_boost * lp + self._out_hf_keep * (x - lp)
        R = self.dc_out.R
        return lfilter([1.0, -1.0], [1.0, -R], tilted)

    # ----------------------------------------------------------------
    def render(self, dur, vel, pedal=False, release_time=None):
        n = int(self.sr * dur)
        buf = np.zeros(n)       # thump/body: raw recorded audio, bypasses output stage
        model = np.zeros(n)     # synthesized model output: gets the final tilt + DC block
        rel = int(self.sr * release_time) if release_time else None
        self.note_on(vel, pedal)
        # Inject sample-based hammer thump at attack — adds the broadband
        # "felt-on-string" thud that pure waveguide string lacks.
        from calib.store import get_calibration as _gc
        thump_gain = float(_gc().get_global('hammer_thump_gain'))
        if thump_gain > 0:
            import os as _os
            thump_path = _os.path.join(_os.path.dirname(__file__),
                                        'calib', 'hammer_thump.npy')
            if _os.path.exists(thump_path):
                thump = np.load(thump_path).astype(np.float64)
                v_norm = (vel / 127.0) ** 1.2  # softer velocity curve for thump
                L = min(len(thump), n)
                buf[:L] += thump[:L] * thump_gain * v_norm
        # Body bass thump — 600 ms low-pass-filtered bass response from real
        # reference, scaled by velocity.  Provides the sub-bass weight the
        # waveguide string can't generate from its f0=261Hz fundamental.
        body_gain = float(_gc().get_global('body_thump_gain'))
        if body_gain > 0:
            import os as _os
            body_path = _os.path.join(_os.path.dirname(__file__),
                                       'calib', 'body_thump.npy')
            if _os.path.exists(body_path):
                body = np.load(body_path).astype(np.float64)
                v_norm = (vel / 127.0) ** 1.2
                L = min(len(body), n)
                buf[:L] += body[:L] * body_gain * v_norm
        for i in range(n):
            if rel is not None and i == rel:
                self.note_off()
            model[i] += self.tick()
            if not self.active:
                break
        # FIR / EQ soundboard: stream was buffered during ticks, convolve now.
        # Mixed into `model` (not `buf`) so it passes through the same final
        # output stage as the direct/IIR path below.
        if isinstance(self.board, (SoundboardFIR, SoundboardEQ)):
            conv = self.board.flush()
            L = min(len(conv), len(model))
            model[:L] = model[:L] + conv[:L] * self._board_mix
        buf += self._apply_output_stage(model)
        gain = params.velocity_gain(self.midi)
        if gain != 1.0:
            buf *= gain
        return buf
