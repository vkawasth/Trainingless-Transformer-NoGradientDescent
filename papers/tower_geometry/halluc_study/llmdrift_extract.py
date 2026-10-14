"""llmdrift_extract.py -- answer-extraction rules for the LLM-drift data (Chen, Zaharia and Zou 2024), copied unchanged from the
validation paper's analysis code so that Study 1 is self-contained.
Data: the <TASK>_EVAL.csv files of https://github.com/lchen001/LLMDrift. Point LLMDRIFT_DIR at the folder that holds them
(default: ./data/llmdrift)."""
import os, re
D = os.environ.get("LLMDRIFT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llmdrift"))
MODELS = {'35M': 'gpt-3.5-turbo-0301', '35J': 'gpt-3.5-turbo-0613', '4M': 'gpt-4-0314', '4J': 'gpt-4-0613'}

def letter(s):
    """The final answer letter: the last 'answer is (X)'; otherwise the parenthesized letter if only one is given, else the last one."""
    s = str(s)
    m = re.findall(r'answer is:?\s*\(?([A-J])\b', s)
    if m: return m[-1]
    m = re.findall(r'\(([A-J])\)', s)
    if m: return m[0] if len(set(m)) == 1 else m[-1]
    m = re.match(r'\s*([A-J])[\.\)]', s); return m.group(1) if m else None
def yesno(s):
    """The final [Yes]/[No] (the last bracketed one), else a leading yes/no."""
    s = str(s); m = re.findall(r'\[(yes|no)\]', s, re.I)
    if m: return m[-1].lower()
    m = re.match(r'\s*(yes|no)\b', s, re.I); return m.group(1).lower() if m else None
def boxed(s):
    m = re.findall(r'\\boxed\{\s*(-?\d+)\s*\}', str(s)); return m[-1] if m else None
def norm_txt(s): return re.sub(r'[^a-z0-9 ]', '', str(s).lower()).strip()
