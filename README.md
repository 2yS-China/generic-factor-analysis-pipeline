# Factor Analysis Pipeline

This package contains everything your client needs to run the **ML–Varimax
factor‑analysis** workflow on any tidy dataset.

## Contents

| File | Purpose |
|------|---------|
| `factor_analysis_pipeline.py` | Main CLI script (Python ≥ 3.8) |
| `requirements.txt` | Pip dependencies |
| `run_factor_analysis.sh` | *nix one‑click wrapper |
| `run_factor_analysis.bat` | Windows one‑click wrapper |

## Quick‑start (all platforms)

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run (replace your_data.xlsx with your file)
python factor_analysis_pipeline.py \
    --input your_data.xlsx \
    --id-col Year \
    --output-prefix results/demo
```

## One‑click wrappers


* **Windows**

  ```bat
  run_factor_analysis.bat your_data.xlsx results\demo
  ```

Each run will create:

* `results/demo_scree.png`
* `results/demo_factor_loadings.csv`
* `results/demo_factor_scores.csv`

## Optional: build a standalone executable (Windows)

```powershell
pip install pyinstaller
pyinstaller -F factor_analysis_pipeline.py
# Output EXE will be in dist/  (≈35 MB)
```

Enjoy!
