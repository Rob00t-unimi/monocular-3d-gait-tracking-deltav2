# monocular-3d-gait-tracking-deltav2 (Gait360): 360-Degree Monocular 3D Gait Analysis via DELTAv2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architecture: DELTAv2](https://img.shields.io/badge/3D%20Tracker-DELTAv2%20%2B%20UniDepthV2-success.svg)](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
[![Tests: 106 Passed](https://img.shields.io/badge/tests-106%20passed-brightgreen.svg)](tests/)

**`monocular-3d-gait-tracking-deltav2`** (**Gait360**) is a markerless monocular 3D gait analysis framework that extracts clinical thigh kinematics (sagittal flexion/extension pitch, 3D linear accelerations, step cadence, and rigid-body stability) across **arbitrary 360° camera viewing angles** (`90°` pure lateral, `45°` front/rear oblique, `0°` frontal approaching, and `180°` posterior receding), validated against synchronized wearable **WitMotion IMU** ground-truth sensors across **40 stratified videos** (39 subjects).

---

## 1. Repository Structure (Zero Root Clutter)

```text
monocular-3d-gait-tracking-deltav2/
├── README.md                          # Project overview, CLI guide & benchmark summary
├── CHANGELOG.md                       # Keep-a-Changelog release history (v0.1.0, v0.5.0, v1.0.0)
├── pyproject.toml                     # Package configuration & `gait360` CLI entrypoint
├── .gitignore                         # Ignores heavy datasets, weights & temporary logs
├── .gitattributes                     # Git LFS tracking rules for .npz and .pdf artifacts
├── configs/
│   ├── default_config.yaml            # Pipeline hyperparameters & paths
│   ├── envs/                          # Conda YAML environments (delta_env, analysis_env, etc.)
│   └── setup/                         # Cross-platform setup & CUDA/DELTAv2 diagnostic scripts
├── data/                              # Manifests, cached states & NTFS Junctions (Gait_Dataset, gt_cache)
├── docs/
│   ├── 01_SYSTEM_ARCHITECTURE_AND_CLI.md
│   ├── 02_STEP_BY_STEP_METHODOLOGY.md
│   ├── 03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md
│   ├── 04_MATHEMATICAL_FORMULATION_AND_FIXES.md
│   ├── 05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md
│   └── references/                    # Reference literature PDFs (DELTAv2, DELTA, SpaTrackerV2, VGGT)
├── examples/                          # 5 End-to-End Interactive Jupyter Notebooks (01..05)
├── experiments/                       # Historical Decision Reports & Prototypes
│   ├── 01_spatracker_vggt/            # Phase 1: VGGT & SpaTrackerV2 bidirectional exploration
│   ├── 02_delta_v1/                   # Phase 2: DELTA v1 dense 3D & anatomical anchor trials
│   └── 03_deltav2_prototyping/        # Phase 3: DELTAv2 validation & 16-video cohort JSON
├── models/                            # External 3D Trackers via NTFS Junctions (DELTA, DELTAv2)
├── results/
│   ├── benchmark_40_metrics.json      # Authoritative 40-video per-trial & aggregate metrics
│   ├── benchmark_40_summary.csv       # Tabular summary across all 4 evaluation tiers
│   ├── REPORT_40_VIDEOS_BENCHMARK.md  # Full 40-video 4-tier benchmark report
│   ├── REPORT_TRIPLET_VS_CLOUD_ABLATION.md
│   └── REPORT_16_VIDEOS_MODEL_COMPARISON.md
├── src/
│   ├── cli.py                         # Unified parametric CLI (`gait360` / `python -m src.cli`)
│   ├── core/                          # Domain models, interfaces, constants & exceptions
│   ├── utils/                         # PEP8 path validator & project root resolver
│   ├── data_io/                       # Video, OpenPose JSON, WitMotion IMU CSV & state I/O
│   ├── preprocessing/                 # 2D thigh vision segmentation & Butterworth filtering
│   ├── inference/                     # BaseTracker, DummyTracker & DELTAv2Tracker
│   ├── postprocessing/                # 3-Stage GeometricPruner & rigid triplet extraction
│   ├── analysis/                      # 3D SVD Kabsch alignment & kinematic signal extraction
│   ├── metrics/                       # Waveform, magnitude, frequency & 3D geometry metrics
│   ├── pipeline/                      # Stage orchestrators (dataset, cohort, vision, track, prune)
│   ├── benchmark/                     # Multi-tier evaluator & deterministic report generator
│   └── ui/                            # 4 Tkinter interactive 2D/3D/waveform inspectors
└── tests/                             # 106 automated unit & integration tests
```

---

## 2. Unified Command-Line Interface (`gait360`)

All pipeline stages and interactive graphical tools are consolidated into `src/cli.py`:

```bash
# 1. Scan multi-view dataset and build candidate quality cache
python -m src.cli scan --mode all

# 2. Select deterministic 40-video stratified cohort (or open GUI with --gui)
python -m src.cli select-cohort

# 3. Run 2D thigh segmentation, reference frame finder & 3-point anchor extraction
python -m src.cli preprocess --config configs/default_config.yaml

# 4. Run dense 3D trajectory inference (deltav2 or dummy) with .npz resume/caching
python -m src.cli track --tracker deltav2

# 5. Run 3-Stage Geometric Pruner (z_weight=0.25, h_min=0.035m, AR<=8.0)
python -m src.cli prune --z-weight 0.25 --min-altitude 0.035 --max-aspect-ratio 8.0

# 6. Compute 3D SVD-aligned kinematic metrics and regenerate JSON, CSV & Markdown reports
python -m src.cli evaluate

# 7. Launch interactive 2D/3D/waveform inspectors (keypoints | cohort | 3d | signals)
python -m src.cli inspect --mode 3d
```

---

## 3. 40-Video SOTA Benchmark Summary

| Metric (40 Stratified Videos) | 1. Dummy Tracker | 2. DELTAv2 Raw | 3. DELTAv2 Pruned Cloud | 4. DELTAv2 Refined Triplet |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.305` | `0.185 ± 0.258` | `0.276 ± 0.278` | **`0.295 ± 0.329`** *(wins 21/40 vs Cloud)* |
| **Pitch Peak XCorr** ($\uparrow$) | `0.369 ± 0.175` | `0.331 ± 0.156` | `0.391 ± 0.169` | **`0.415 ± 0.195`** |
| **Pitch nRMSE (%)** ($\downarrow$) | `33.2 ± 9.7` | `30.8 ± 6.7` | `34.4 ± 10.6` | **`30.5 ± 9.3`** |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7% ± 88.2%` | `-15.5% ± 66.2%` | `+45.2% ± 112.9%` | **`+4.1% ± 70.8%`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.163` | `0.243 ± 0.168` | **`0.273 ± 0.152`** | `0.255 ± 0.162` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.345 ± 0.328` | `1.242 ± 0.168` | `1.239 ± 0.160` | **`1.230 ± 0.173`** |
| **3D Rigid Deformation (%)** ($\downarrow$) | `16.4 ± 9.4` *(2D)* | `110.9 ± 26.8` | `41.4 ± 20.1` | **`28.4 ± 9.9`** *(wins 39/40 = 97.5% vs Cloud)* |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 592.3` | `179.9 ± 62.7` | **`28.9 ± 11.4`** | `36.6 ± 14.3` |
| **True Identity Swaps ($90^\circ$ & $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `3 / 30 (10.0%)` | **`3 / 30 (10.0%)`** |
| **Axial Depth Artifacts ($0^\circ$ & $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `1 / 10 (10.0%)` | **`1 / 10 (10.0%)`** |

---

## 4. Documentation & Interactive Tutorials

- **Technical Documentation (`docs/`):**
  1. [01 — System Architecture, State Management & CLI](docs/01_SYSTEM_ARCHITECTURE_AND_CLI.md)
  2. [02 — Step-by-Step Pipeline Methodology](docs/02_STEP_BY_STEP_METHODOLOGY.md)
  3. [03 — Experimental Chronicle & 3D Tracking Models](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
  4. [04 — Mathematical Formulation & Critical Fixes](docs/04_MATHEMATICAL_FORMULATION_AND_FIXES.md)
  5. [05 — Benchmark Results & Ablation Study](docs/05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md)
- **End-to-End Jupyter Notebooks (`examples/`):**
  1. [`01_dataset_scanning_and_imu_loading.ipynb`](examples/01_dataset_scanning_and_imu_loading.ipynb)
  2. [`02_thigh_segmentation_and_anchor_selection.ipynb`](examples/02_thigh_segmentation_and_anchor_selection.ipynb)
  3. [`03_deltav2_dense_3d_tracking.ipynb`](examples/03_deltav2_dense_3d_tracking.ipynb)
  4. [`04_geometric_pruning_and_rigid_triplet.ipynb`](examples/04_geometric_pruning_and_rigid_triplet.ipynb)
  5. [`05_kinematic_alignment_and_benchmark_metrics.ipynb`](examples/05_kinematic_alignment_and_benchmark_metrics.ipynb)
