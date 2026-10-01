# AllSides: big events and phase changes (2015–2020)

Data (not redistributed): Baly et al. 2020, https://github.com/ramybaly/Article-Bias-Prediction
```
git clone --depth 1 https://github.com/ramybaly/Article-Bias-Prediction /path/abp     # ~665 MB, ~10 s
export ALLSIDES_DIR=/path/abp/data/jsons      # or: ln -s /path/abp/data examples/allsides/data
pip install vaderSentiment
```
| command (from `examples/allsides/`) | what it reproduces | output |
|---|---|---|
| `python load.py` | lexicon markers per article (adversarial, directed, honorific, tone); builds `allsides_cache.pkl` (≈ 20 s) | summary by side |
| `python events.py` | six pre-specified events: outcome probabilities, L−R gap, directed gap, naturality Wald test, Slepian step, multitaper band ratios, line F (≈ 15 s) | `events_out.txt`, `events_results.json` |
| `python scan.py` | unrestricted Slepian-step scan of the weekly gap, block-bootstrap p (≈ 1 min) | `scan_out.txt`, `scan_results.json` |

Synthetic check of the spectral detectors: `cd ../dynamics && python spectral_synthetic.py` (≈ 2 min).
Delete `allsides_cache.pkl` after changing the lexicons in `load.py`.
