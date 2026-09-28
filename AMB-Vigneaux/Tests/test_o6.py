import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(__file__)
B = os.path.join(HERE, "..", "benchmarks", "o6")


def test_generator_and_scorer_roundtrip(tmp_path):
    out = tmp_path / "c"
    subprocess.run([sys.executable, os.path.join(B, "o6_generate.py"), "--out", str(out), "--depth", "3",
                    "--nsym", "6", "--nleaf", "9", "--nrules", "3", "--reverse-valid", "--n-train", "400",
                    "--n-val", "50", "--n-test", "60", "--seed", "1"], check=True, capture_output=True)
    sys.path.insert(0, B)
    import o6_score
    PT = o6_score.truth_tables(str(out))
    np.savez(tmp_path / "t.npz", **{f"P{l}": PT[l] for l in PT})
    r = subprocess.run([sys.executable, os.path.join(B, "o6_score.py"), "--data", str(out), "--model",
                        str(tmp_path / "t.npz")], check=True, capture_output=True, text=True)
    s = json.loads(r.stdout)
    assert abs(s["gap"]) < 1e-9                                   # the truth scores as the truth
    assert s["levels"] == s["levels_truth"]
    lat = json.load(open(out / "test_latents.json"))
    assert len(lat) == 60 and [len(x) for x in lat[0]] == [4, 2, 1]
