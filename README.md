# AU-Fukaya Compiler Tests

**Patent**: 64/092,381 · 64/092,056 · 64/085,268 · 64/085,273 · 64/090,029  
**GitHub**: https://github.com/vkawasth/Trainingless-Transformer-NoGradientDescent

## Quick run (no corpus needed, ~1s)
```bash
python -m pytest tests/test_compiler.py::TestConfirmedResults tests/test_compiler.py::TestFukayaCategory -v
```

## Full run (requires corpus at /tmp/*.json, ~5 min)
```bash
python build_corpus.py --out /tmp/ --loops 300
python -m pytest tests/test_compiler.py -v
```

## Test groups

| Group | Tests | Corpus? | Time |
|-------|-------|---------|------|
| TestCorpusStatistics | Sparsity, entropy, GT invariant | yes | fast |
| TestSpectralEmbedding | Pass 0: E₀ shape, std, next-token gap | yes | fast |
| TestK0Structure | Emb dominance, K₁ attractor, orthogonality | yes | ~30s |
| TestPass12 | Pass 12: pre-baked + 25CE | yes | ~60s |
| TestConfirmedResults | Regression table (no training) | no | <1s |
| TestFukayaCategory | Strip areas, m₂ wall detection | no | <1s |

## Confirmed results locked in regression table

| Experiment | val | status |
|-----------|-----|--------|
| Spectral E₀ init | 4.46 | ✓ |
| Pre-baked + 25 CE | 3.44 | ✓ |
| Pass 12 (26 steps) | 2.54 | ✓ |
| K₀ split 6×(25+LM) | 0.139 | ✓ |
| 167 plain CE | 0.999 | ✓ |
| Compiler + 167 CE | 0.095 | ✓ |
| Teacher (24L, 300 CE) | 0.250 | reference |

Run compiler_geometric.py 

<img width="370" height="702" alt="Screenshot 2026-06-22 at 6 32 03 PM" src="https://github.com/user-attachments/assets/776af35f-a7e7-44aa-97fa-977c6fa488c3" />

Full Demo : python compiler_demo_2.py --no_baseline

<img width="536" height="738" alt="Screenshot 2026-06-21 at 8 01 19 AM" src="https://github.com/user-attachments/assets/150d841d-b5df-4bf4-bf54-81d9ee668742" />

Path Profilers 

<img width="480" height="750" alt="Screenshot 2026-06-21 at 8 09 25 AM" src="https://github.com/user-attachments/assets/f9e23c87-742e-4905-8d0b-4be0302f3679" />

<img width="458" height="753" alt="Screenshot 2026-06-21 at 8 12 28 AM" src="https://github.com/user-attachments/assets/447111d0-08ff-46bc-a99f-38a25d28ddb1" />




