# 01 — System Architecture, State Management, and Unified CLI (`gait360`)

## 1. Architectural Overview

`monocular-3d-gait-tracking-deltav2` (**Gait360**) is structured into seven top-level functional directories:

```text
monocular-3d-gait-tracking-deltav2/
├── README.md                          # Project overview, CLI guide & benchmark summary
├── CHANGELOG.md                       # Keep-a-Changelog release history (v0.1.0, v0.5.0, v1.0.0)
├── pyproject.toml                     # Package configuration & `gait360` CLI entrypoint
├── .gitignore                         # Ignores heavy datasets, weights & temporary logs
├── .gitattributes                     # Git LFS tracking rules for .npz and .pdf artifacts
├── index.html                         # Interactive HTML documentation & bilingual viewer (GitHub Pages)
├── configs/
│   ├── default_config.yaml            # Pipeline hyperparameters & paths
│   ├── envs/                          # Conda YAML environments (delta_env, analysis_env, etc.)
│   └── setup/                         # Cross-platform setup & CUDA/DELTAv2 diagnostic scripts
├── data/                              # Manifests, cached states, datasets (Gait_Dataset) & IMU cache (gt_cache)
├── docs/
│   ├── 01_SYSTEM_ARCHITECTURE_AND_CLI.md
│   ├── 02_STEP_BY_STEP_METHODOLOGY.md
│   ├── 03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md
│   ├── 04_MATHEMATICAL_FORMULATION_AND_FIXES.md
│   ├── 05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md
│   ├── 06_DUMMY_BASELINE_METHOD.md
│   ├── CONCLUSIONS.md                 # Executive summary: DELTAv2 vs Dummy across 4 key indicators
│   ├── docs_it/                       # Italian documentation mirror (README, chapters, reports)
│   └── references/                    # Reference literature PDFs (DELTAv2, DELTA, SpaTrackerV2, VGGT)
├── examples/                          # 5 End-to-End Interactive Jupyter Notebooks (01..05)
├── experiments/                       # Historical Decision Reports & Prototypes
│   ├── 01_spatracker_vggt/            # Phase 1: VGGT & SpaTrackerV2 bidirectional exploration
│   ├── 02_delta_v1/                   # Phase 2: DELTA v1 dense 3D & anatomical anchor trials
│   └── 03_deltav2_prototyping/        # Phase 3: DELTAv2 validation & 16-video cohort JSON
├── models/                            # External 3D Trackers & submodules (DELTA, DELTAv2, SpaTrackerV2, VGGT)
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

## 2. Caching, State Persistence

1. **Multi-View Trial Alignment Cache (`data/cached_dataset_map.json`):** Serializes the mapping between RGB videos (`90°`, `45°`, `Front`, `Rear`), OpenPose 2D keypoint directories, and WitMotion IMU ground-truth CSV files across all 59 subjects.
2. **Candidate Video Quality Cache (`data/candidate_videos_cache.json`):** Stores pre-evaluated IMU ground-truth quality scores (`Optimal`, `Good`, `Imprecise`), Range of Motion (ROM), and step cadence for rapid stratified cohort selection.
3. **2D Vision Keypoint State (`data/videos_state.json`):** Persists the selected reference frame index (`ref_frame`), walk start frame (`walk_start`), and the 3 anatomical thigh query points (`p1`, `p2`, `p3`) with automatic resume support.
4. **3D Trajectory Tensor Cache (`data/inference_results/raw/*.npz` & `refined/*.npz`):** Stores raw DELTAv2 `(T, N, 3)` metric point clouds and geometrically pruned point clouds + rigid triplets `(T, 3, 3)`.

## 3. Unified Command-Line Interface (`gait360` / `python -m src.cli`)

All pipeline stages and interactive inspectors are accessible via a single parametric entrypoint:

```bash
# 0. Verify dataset integrity and restructure raw files against manifest
python -m src.cli align-dataset --dry-run

# 1. Scan dataset and build candidate quality cache
python -m src.cli scan --mode all

# 2. Select deterministic 40-video stratified benchmark cohort (or open GUI with --gui)
python -m src.cli select-cohort

# 3. Run 2D vision thigh segmentation & 3-point anchor extraction
python -m src.cli preprocess --config configs/default_config.yaml

# 4. Run dense 3D point tracking (deltav2 or dummy) with automatic .npz resume
python -m src.cli track --tracker deltav2

# 5. Run 3-Stage Geometric Pruner (z_weight=0.25, h_min=0.035m, AR<=8.0)
python -m src.cli prune --z-weight 0.25 --min-altitude 0.035 --max-aspect-ratio 8.0

# 6. Compute 3D SVD-aligned kinematic metrics and regenerate JSON, CSV & Markdown reports
python -m src.cli evaluate

# 7. Launch interactive GUIs (keypoints | cohort | 3d | signals)
python -m src.cli inspect --mode 3d
```
