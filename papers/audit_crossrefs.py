"""audit_crossrefs.py -- check every numbered cross-paper citation against the target paper's .aux/.toc."""
import re, sys
P = "/home/claude/paper/"
papers = {"theory": "conditionals_couplings", "companion": "coupling_polytopes_companion", "tower": "three_level_tower_v10",
          "layers": "/home/claude/tl/build/tl", "attrib": "/home/claude/pa/build/pa"}
keys = {"theory": {"Awasthi-Tower": "tower", "Awasthi-Family": "companion", "Awasthi-ThreeLayers": "layers"},
        "companion": {"Conditionals": "theory", "Tower": "tower", "ThreeLayers": "layers"},
        "tower": {"Conditionals": "theory", "Awasthi-Family": "companion", "Awasthi-ThreeLayers": "layers"},
        "layers": {"awasthiC": "theory", "awasthiF": "companion", "awasthiT": "tower"},
        "attrib": {"Awasthi-Conditionals": "theory", "Awasthi-Family": "companion", "Awasthi-Tower": "tower", "Awasthi-ThreeLayers": "layers"}}
keys["theory"]["Awasthi-Attribution"] = "attrib"; keys["tower"]["Awasthi-Attribution"] = "attrib"; keys["companion"]["Attribution"] = "attrib"
def labels(name):
    aux = open((papers[name] if papers[name].startswith("/") else P + papers[name]) + ".aux").read(); out = {}
    for lab, num in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux): out.setdefault(num, set()).add(lab.split(":")[0])
    toc = open((papers[name] if papers[name].startswith("/") else P + papers[name]) + ".toc").read()
    for num in re.findall(r"\\numberline \{([^}]*)\}", toc): out.setdefault(num, set()).add("sec")
    return out
L = {k: labels(k) for k in papers}
KIND = {"Thm": {"thm"}, "Prop": {"prop"}, "Def": {"def"}, "Cor": {"cor"}, "Obs": {"obs"}, "Question": {"q", "conj"},
        "Conj": {"conj"}, "Conj.": {"conj"}, "Lemma": {"lem"}, "Lem": {"lem"}, "Ex": {"ex"}, "Remark": {"rem"}, "Section": {"sec", "app", "subsec"}, "S": {"sec", "app"},
        "App": {"sec", "app"}}
bad = 0; total = 0
for src, km in keys.items():
    tex = open((papers[src] if papers[src].startswith("/") else P + papers[src]) + ".tex").read()
    for opt, key in re.findall(r"\\cite\[([^\]]*)\]\{([^}]*)\}", tex):
        if key not in km: continue
        tgt = km[key]
        lastkind = None
        for item in re.split(r",\s*", opt):
            item = item.replace("~", " ").replace("\\S\\S", "S ").replace("\\S", "S ").strip()
            item = re.sub(r"^(Thm|Prop|Cor|Def|Lem|Obs)s\.", r"\1.", item)
            if re.match(r"^[0-9]", item) and lastkind: item = lastkind + ". " + item
            m = re.match(r"(Thm|Prop|Def|Cor|Obs|Question|Conj|Lemma|Lem|Ex|Remark|Section|Sections|App|S)\.?\s*([0-9A-Z][0-9.]*)(?:--([0-9]+))?(\([ivx]+\))?", item)
            if not m:
                if re.match(r"^S?\s*[0-9]", item):
                    m = re.match(r"()S?\s*([0-9][0-9.]*)()()", item)
                else:
                    print(f"  ? unparsed in {src}: [{opt}]{{{key}}}"); continue
            kind, num = m.group(1).rstrip("s") or "S", m.group(2).rstrip(".")
            lastkind = kind if kind not in ("S", "Section", "App") else None
            if kind == "Section": kind = "Section"
            nums = [num] + ([m.group(3)] if m.group(3) else [])
            for nm in nums:
                total += 1; kinds = L[tgt].get(nm, set()); want = KIND.get(kind, set())
                ok = bool(kinds & want) if want else bool(kinds)
                if not ok:
                    bad += 1; print(f"  MISMATCH {src} -> {tgt}: [{opt}] item '{item}' -> number {nm} has {sorted(kinds) or 'nothing'}")
print(f"{total} numbered cross-paper references checked, {bad} mismatches")
