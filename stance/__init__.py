"""Indic Climate Stance: stance detection for Hindi/Bengali social-media text toward
"Climate change and global warming is a serious concern"."""
import sys

# Windows consoles default to cp1252 and crash on Devanagari/Bengali output.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
