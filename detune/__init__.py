"""Piano detuning analyzer.

Estimates the integral degree of detuning of a piano in a recording, combining:
  * unison detuning  — how far the 2-3 strings of each note drift apart (beating)
  * global tuning    — how consistently note pitches sit on a self-consistent scale

See detune/analyze.py for the CLI entry point:  python -m detune.analyze --wav <file>
"""
