"""Theorem numbers for figure text, read from the LaTeX .aux file so that figures follow renumbering. Usage: from auxref import N; N("thm:history")."""
import os, re
_AUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditioned_states_kahler.aux")
_TAB = {}
if os.path.exists(_AUX):
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(_AUX, encoding="utf-8", errors="ignore").read()): _TAB.setdefault(m.group(1), m.group(2))
def N(label): return _TAB.get(label, "??")
