# Geometric Invariance Theory & More structured corpus.

<img width="1408" height="768" alt="Gemini_Generated_Image_ckosfuckosfuckos" src="https://github.com/user-attachments/assets/970ad1f7-4820-498c-bcde-dd840438d5e8" />


<img width="1408" height="768" alt="Gemini_Generated_Image_gqn0uygqn0uygqn0" src="https://github.com/user-attachments/assets/d03c41ae-f121-4fb8-ae76-9768f99905ab" />


<img width="456" height="398" alt="Screenshot 2026-09-13 at 6 50 07 AM" src="https://github.com/user-attachments/assets/537b2a27-ff99-4ae4-8c4b-9d1959c93b11" />

# Morse-Smale Witness complex

<img width="703" height="577" alt="Screenshot 2026-09-13 at 11 44 22 AM" src="https://github.com/user-attachments/assets/b3bf0398-5d27-47bc-8e60-8bea9be58a8a" />

<img width="3300" height="2550" alt="zone3_witness_final_clean" src="https://github.com/user-attachments/assets/47351fd4-fa7f-43bb-a033-83a5058b4dad" />

# Drastic reduction in Geometric compiler steps using Smale-Morse Loss complex.

<img width="597" height="275" alt="Screenshot 2026-09-13 at 1 51 41 PM" src="https://github.com/user-attachments/assets/58917ee1-facb-4401-b0d7-96ba786bbe62" />


<img width="377" height="590" alt="Screenshot 2026-09-13 at 1 52 17 PM" src="https://github.com/user-attachments/assets/84c06f94-7f61-4ae3-be4c-0f2f174baf57" />


<img width="369" height="169" alt="Screenshot 2026-09-13 at 1 52 51 PM" src="https://github.com/user-attachments/assets/b4f1584c-6bf1-427c-a25c-35bd82c73391" />






# Cxx23 Phased Geometric Compiler with ops.
Since 5 phase Geometric compiler knows in advance where floor is going to be, we can separate phases and 
only rely on Geometric indicators (unlike driving "loss" down) to conclude weight setting. We consider 
Transformer weight setting a Gradient Flow Problem (not an optimization problem) so we can finish these
Phases separately as compilers farm regions of compilation. We keep separation of concern at C++ level.

# Functioning Distributed Training

<img width="585" height="449" alt="Screenshot 2026-07-09 at 6 01 20 PM" src="https://github.com/user-attachments/assets/d9a8b04b-e71d-4a1d-89b2-1a8956936020" />



# Build and Run
Go to PhasedCompiler directory and run ./build_all.sh run

To only run python side - python compiler_geometri_patched_86.py  # Runs 5 phases in sequential fashion

To only run python with memory profiling as reported in paper ver 7 (latest, others are there for context) :

       run python compiler_geometri_patched_86_memfixed.py        # (profiles fwd pass/bwd pass memory utilization)
       

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

Snappers Lemma based Jumps instead of slowly walking via Gradient Descent

<img width="841" height="1873" alt="Screenshot 2026-07-05 at 4 16 02 PM" src="https://github.com/user-attachments/assets/79873d09-8e4b-4e33-bf10-5cae59246bdc" />

Memory Saving

<img width="2800" height="1200" alt="memory_viz" src="https://github.com/user-attachments/assets/55f82f29-a24b-48ff-9341-fb9ad70dbece" />



Run : Uses only Geometry to decide phases --  compiler_geometric.py 

<img width="370" height="702" alt="Screenshot 2026-06-22 at 6 32 03 PM" src="https://github.com/user-attachments/assets/776af35f-a7e7-44aa-97fa-977c6fa488c3" />

Full Demo : python compiler_demo_2.py --no_baseline

<img width="536" height="738" alt="Screenshot 2026-06-21 at 8 01 19 AM" src="https://github.com/user-attachments/assets/150d841d-b5df-4bf4-bf54-81d9ee668742" />

Path Profilers 

<img width="480" height="750" alt="Screenshot 2026-06-21 at 8 09 25 AM" src="https://github.com/user-attachments/assets/f9e23c87-742e-4905-8d0b-4be0302f3679" />

<img width="458" height="753" alt="Screenshot 2026-06-21 at 8 12 28 AM" src="https://github.com/user-attachments/assets/447111d0-08ff-46bc-a99f-38a25d28ddb1" />




