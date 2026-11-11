# monocular-3d-gait-tracking-deltav2 (Gait360): 360-Degree Monocular 3D Gait Analysis via DELTAv2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architecture: DELTAv2](https://img.shields.io/badge/3D%20Tracker-DELTAv2%20%2B%20UniDepthV2-success.svg)](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
[![Tests: 106 Passed](https://img.shields.io/badge/tests-106%20passed-brightgreen.svg)](tests/)

**`monocular-3d-gait-tracking-deltav2`** (**Gait360**) is a markerless monocular 3D gait analysis framework that extracts clinical thigh kinematics (sagittal flexion/extension pitch, 3D linear accelerations, step cadence, and rigid-body stability) across **arbitrary 360° camera viewing angles** (`90°` pure lateral, `45°` front/rear oblique, `0°` frontal approaching, and `180°` posterior receding), validated against synchronized wearable **WitMotion IMU** ground-truth sensors across **40 stratified videos** (39 subjects).

---

## 1. Repository Structure

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

## 2. Unified Command-Line Interface (`gait360`)

The entire pipeline is driven by a single parametric command-line interface, accessible either via the installed executable `gait360` (after `pip install -e .`) or directly via Python (`python -m src.cli <command> [options]`).

```text
usage: gait360 [-h] {scan,select-cohort,preprocess,track,prune,evaluate,inspect} ...
```

---

### Command Reference & Parameters

#### 1. `scan` — Multi-Modal Dataset Mapping & Quality Cache
Scans the multi-view dataset directory, matches OpenPose 2D keypoints with synchronized WitMotion IMU CSV sensors, and builds the quality candidate cache.
- **Syntax:** `python -m src.cli scan [options]`
- **Parameters:**
  - `--mode {all, scan, candidates}` *(default: `all`)*:
    - `scan`: scans folder structure and generates `data/cached_dataset_map.json`.
    - `candidates`: evaluates candidate video signal quality and generates `data/candidate_videos_cache.json`.
    - `all`: runs both scanning and candidate quality evaluation.
  - `--dataset-dir <path>` *(default: `data/Gait_Dataset`)*: Path to the raw multi-view dataset folder.
  - `--cache-path <path>` *(default: `data/cached_dataset_map.json`)*: Output path for the serialized dataset map JSON.
  - `--force`: Force rebuild of candidate quality cache even if `data/candidate_videos_cache.json` already exists.
- **Examples:**
  ```bash
  python -m src.cli scan                         # Reuses existing cache (<1s)
  python -m src.cli scan --force                 # Recomputes candidate quality cache across all trials
  python -m src.cli scan --mode scan             # Re-indexes only folder hierarchy
  ```

---

#### 2. `select-cohort` — Stratified 40-Video Benchmark Selection
Deterministically selects the balanced, stratified 40-video benchmark cohort across all 8 viewing perspectives (5 videos per class), guaranteeing subject uniqueness (40 unique participants) and valid leg sensor placements.
- **Syntax:** `python -m src.cli select-cohort [options]`
- **Parameters:**
  - `--gui`: Launches the interactive Tkinter Stratified Cohort Selector GUI (`InferenceVideoSelectorGUI`) featuring category tabs, quality rankings, uniqueness constraints, and direct video playback. Without this flag, runs deterministically in headless CLI mode.
  - `--force`: Force regeneration of `selected_inference_videos_40.csv` and `.txt` even if cached manifests already exist.
- **Generated Artifacts:**
  - `data/selected_inference_videos_40.csv`
  - `data/selected_inference_videos_40.txt`
- **Examples:**
  ```bash
  python -m src.cli select-cohort                # Deterministic headless CLI export
  python -m src.cli select-cohort --gui          # Opens interactive Tkinter selection GUI
  ```

---

#### 3. `preprocess` — 2D Vision Segmentation & Reference Frame Selection
Performs 2D vision preprocessing: detects subject silhouettes using MediaPipe, identifies the terminal swing maximum horizontal knee spread peak ($|x_{\text{RKnee}} - x_{\text{LKnee}}|$) as the optimal reference frame, and extracts the 3-point anatomical femur anchor triangle.
- **Syntax:** `python -m src.cli preprocess [options]`
- **Parameters:**
  - `--config <path>` *(default: `configs/default_config.yaml`)*: Path to pipeline YAML configuration.
  - `--gui`: Launches the interactive Matplotlib validation GUI to manually inspect/reposition anchor points.
  - `--video <name>` *(default: `None`)*: Specific video name to inspect in GUI mode (e.g. `P27_E2T2_90.mp4`).
  - `--force`: Force re-preprocessing of IMU ground-truth cache and 2D anchor states (`videos_state.json`).
- **Default Behavior:** Headless automated execution with fast resume (`--auto --resume`). Automatically skips already validated videos in `data/videos_state.json` in <0.2s.
- **Examples:**
  ```bash
  python -m src.cli preprocess                   # Fast automated run with resume (<0.2s if cached)
  python -m src.cli preprocess --gui --video P27_E2T2_90.mp4  # Inspect anchor points for a video in GUI
  ```

---

#### 4. `track` — Dense 3D Point Tracking Inference
Executes dense 3D point tracking inference using the chosen tracker architecture (`deltav2` or `dummy`), storing raw 3D trajectory tensors in `data/inference_results/raw/`.
- **Syntax:** `python -m src.cli track [options]`
- **Parameters:**
  - `--tracker {deltav2, dummy}` *(default: `deltav2`)*:
    - `deltav2`: Production pipeline combining UniDepthV2 metric depth estimation with DELTAv2 coarse-to-fine optical tracking and anatomical silhouette masking.
    - `dummy`: 2D OpenPose pseudo-3D lifted baseline (ablation benchmark).
  - `--video <name>` *(default: `None`)*: Specific video filename to track. If omitted, runs batch inference across all 40 benchmark videos.
  - `--force`: Re-run inference even if cached raw `.npz` trajectory tensor already exists.
- **Examples:**
  ```bash
  python -m src.cli track --tracker deltav2       # Batch run on 40 videos with automatic .npz resume
  python -m src.cli track --tracker deltav2 --video P27_E2T2_90.mp4  # Track single video
  python -m src.cli track --tracker dummy         # Run dummy baseline tracker
  python -m src.cli track --force                # Force re-tracking, overwriting cached .npz files
  ```

---

#### 5. `prune` — 3-Stage Geometric Post-Processing & Triplet Selection
Executes the 3-Stage Geometric Post-Processing pipeline on raw DELTAv2 trajectories to filter out background floaters, dampen monocular depth z-jitter, and extract the optimal rigid triplet protected against normal-vector twist.
- **Syntax:** `python -m src.cli prune [options]`
- **Parameters:**
  - `--video <name>` *(default: `None`)*: Specific video filename to prune. If omitted, processes all raw `.npz` tensors in batch.
  - `--z-weight <float>` *(default: `0.25`)*: Anisotropic depth axis weighting for Stage 1-2 spatial DBSCAN and MAD velocity filtering.
  - `--min-altitude <float>` *(default: `0.035`)*: Minimum physical triangle altitude ($h_{\min}$) in meters to guard against collinear triplet twist.
  - `--max-aspect-ratio <float>` *(default: `8.0`)*: Maximum triangle aspect ratio threshold ($\text{AR} = b / h$).
  - `--force`: Force re-pruning even if cached refined `.npz` already exists in `data/inference_results/refined/`.
- **Generated Artifacts:** Refined trajectory tensors in `data/inference_results/refined/`.
- **Examples:**
  ```bash
  python -m src.cli prune                        # Batch pruning across all 40 videos with default guards
  python -m src.cli prune --video P27_E2T2_90.mp4 # Prune single video
  python -m src.cli prune --z-weight 0.20 --min-altitude 0.040
  ```

---

#### 6. `evaluate` — SVD Kinematic Alignment & Benchmark Reporting
Aligns 3D visual trajectories with WitMotion IMU ground-truth coordinate systems using Kabsch SVD rotation, extracts clinical gait waveforms (Pitch, 3D accelerations), computes 4-tier benchmark metrics (Pearson $r$, Peak XCorr, nRMSE, ROM error, DTW, Rigidity, Jerk), and generates authoritative reports.
- **Syntax:** `python -m src.cli evaluate [options]`
- **Parameters:**
  - `--reports-only`: Regenerates markdown reports and summary CSV directly from existing `results/benchmark_40_metrics.json` without recomputing SVD kinematic alignment.
  - `--force`: Force full recomputation of SVD kinematic alignment and metrics across all 40 videos even if cached results exist.
- **Generated Artifacts:**
  - `results/benchmark_40_metrics.json`
  - `results/benchmark_40_summary.csv`
  - `results/REPORT_40_VIDEOS_BENCHMARK.md`
  - `results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`
  - `results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`
- **Examples:**
  ```bash
  python -m src.cli evaluate                     # Full kinematic alignment and benchmark computation
  python -m src.cli evaluate --reports-only      # Instant regeneration of Markdown reports (<1s)
  ```

---

#### 7. `inspect` — Interactive Graphical Visualization Suites
Launches standalone interactive graphical inspectors for qualitative validation of keypoints, cohort balance, 3D point cloud trajectories, or kinematic waveforms.
- **Syntax:** `python -m src.cli inspect [options]`
- **Parameters:**
  - `--mode {keypoints, cohort, 3d, signals, dashboard}` *(default: `3d`)*:
    - `keypoints`: Opens the 2D anatomical anchor and reference frame inspector GUI.
    - `cohort`: Opens the interactive Stratified Cohort Selector Tkinter GUI.
    - `3d`: Opens the interactive 3D Trajectory Inspector (real-time synchronized 3D skeleton + IMU signals).
    - `signals` / `dashboard`: Opens the interactive 3D & 1D Benchmark Comparison Dashboard (`GaitBenchmarkDashboard`) with source toggles (`Triplet`, `Cluster`, `Raw`).
  - `--video <name>` *(default: `P27_E2T2_90.mp4`)*: Specific video filename for `keypoints` mode.
  - `--model {deltav2, spatracker, compare}` *(default: `deltav2`)*: Model mode for `signals`/`dashboard` inspector (`deltav2` for 40-video cohort, `spatracker` or `compare` for 6-video cross-model comparison).
  - `--start <int>` *(default: `0`)*: Starting video index (0-based) for `signals`/`dashboard` mode.
- **Examples:**
  ```bash
  python -m src.cli inspect --mode 3d            # 3D trajectory cloud visualizer
  python -m src.cli inspect --mode signals       # Waveform signal viewer (Vision vs IMU)
  python -m src.cli inspect --mode cohort        # Cohort selection GUI
  python -m src.cli inspect --mode keypoints     # 2D anchor validator GUI
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

---

## 4. Documentation & Interactive Tutorials

- **Technical Documentation (`docs/`):**
  1. [01 — System Architecture, State Management & CLI](docs/01_SYSTEM_ARCHITECTURE_AND_CLI.md)
  2. [02 — Step-by-Step Pipeline Methodology](docs/02_STEP_BY_STEP_METHODOLOGY.md)
  3. [03 — Experimental Chronicle & 3D Tracking Models](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
  4. [04 — Mathematical Formulation & Critical Fixes](docs/04_MATHEMATICAL_FORMULATION_AND_FIXES.md)
  5. [05 — Benchmark Results & Ablation Study](docs/05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md)
  6. [06 — Dummy Baseline Method & Synthetic Ablations](docs/06_DUMMY_BASELINE_METHOD.md)
  7. [Italian Documentation Mirror (`docs/docs_it/`)](docs/docs_it/)
- **End-to-End Jupyter Notebooks (`examples/`):**
  1. [`01_dataset_scanning_and_imu_loading.ipynb`](examples/01_dataset_scanning_and_imu_loading.ipynb)
  2. [`02_thigh_segmentation_and_anchor_selection.ipynb`](examples/02_thigh_segmentation_and_anchor_selection.ipynb)
  3. [`03_deltav2_dense_3d_tracking.ipynb`](examples/03_deltav2_dense_3d_tracking.ipynb)
  4. [`04_geometric_pruning_and_rigid_triplet.ipynb`](examples/04_geometric_pruning_and_rigid_triplet.ipynb)
  5. [`05_kinematic_alignment_and_benchmark_metrics.ipynb`](examples/05_kinematic_alignment_and_benchmark_metrics.ipynb)

---

## 5. Dataset, Submodules & Upstream References

> [!NOTE]
> **DELTAv2 Model & Weights Provenance:**  
> DELTAv2 (*DenseTrack3Dv2*) is a cutting-edge research prototype developed by Snap Research (Ngo et al., *"DELTAv2: Dense Efficient Long-range 3D Tracking for Any video"*, arXiv:2508.01170) as an evolution of DELTA (CVPR 2024). It is not distributed as a package via PyPI. The model architecture was integrated directly from Snap Research's official experimental repository ([`snap-research/DenseTrack3Dv2`](https://github.com/snap-research/DenseTrack3Dv2)), and the official pre-trained checkpoint (`densetrack3dv2.pth`, 205.7 MB) was obtained via the authors' Google Drive mirror (`1Qa9YFAjBFIzrrHHWf8NZMln5YkLfw4Qa`) documented in their repository. DELTAv2 was integrated into this framework on **October 20, 2026** (Phase 3) to overcome the temporal asymmetry and identity swaps observed with SpaTrackerV2 and DELTA v1.

| Component / Submodule | Role in Framework | Scientific Paper & References | Upstream Code & Resources |
| :--- | :--- | :--- | :--- |
| **Gait360** | Multi-View Dataset & Wearable IMU Ground-Truth | Synchronized 360° gait recordings across multiple viewing perspectives (`0°`, `45°`, `90°`, `180°`) and elevations, paired with synchronized 9-DoF **WitMotion IMUs**. Publicly released by Liverpool John Moores University (LJMU). | [LJMU OpenData (eprint/133)](https://opendata.ljmu.ac.uk/id/eprint/133/) |
| **DELTAv2** | Primary Dense 3D Point Tracker | Ngo et al., *"DELTAv2: Dense Efficient Long-range 3D Tracking for Any video"*, arXiv:2508.01170. SOTA joint tracking and dense depth estimation. | [GitHub (`snap-research/DenseTrack3Dv2`)](https://github.com/snap-research/DenseTrack3Dv2) • [Local PDF](docs/references/DELTAv2.pdf) |
| **DELTA** | Phase 2 Baseline 3D Tracker (v1) | Babiloni et al., *"DELTA: Dense Efficient Long-range 3D Tracking"*, CVPR 2024. Initial 3D dense tracker evaluated in Phase 2. | [GitHub (`snap-research/DELTA_densetrack3d`)](https://github.com/snap-research/DELTA_densetrack3d) • [Local PDF](docs/references/DELTA.pdf) |
| **UniDepth / UniDepthV2** | Monocular Metric Depth Backbone | Piccinelli et al., *"UniDepth: Universal Monocular Metric Depth Estimation"*, CVPR 2024. Powers true metric unprojection in DELTAv2. | [GitHub (`lpiccinelli-eth/UniDepth`)](https://github.com/lpiccinelli-eth/UniDepth) • [arXiv:2403.18913](https://arxiv.org/abs/2403.18913) |
| **SpaTracker** | Phase 1 Exploratory 3D Tracker (v1) | Xiao et al., *"Spatial-Temporal Point Tracking"*, 2024. Evaluated during early bidirectional point tracking benchmarking. | [GitHub (`henry123-boy/SpaTracker`)](https://github.com/henry123-boy/SpaTracker) |
| **SpaTrackerV2** | Phase 1 Exploratory 3D Tracker (v2) | Xiao et al., *"Spatial-Temporal Point Tracking v2"*, 2024. Extended point tracker evaluated on multi-scenario gait sequences. | [GitHub (`henry123-boy/SpaTrackerV2`)](https://github.com/henry123-boy/SpaTrackerV2) • [Local PDF](docs/references/SpaTrackerv2.pdf) |
| **VGGT** | Phase 1 Transformer Tracker | Xu et al., *"Visual Geometry Grounded Transformer for Point Tracking"*, 2024. Evaluated for pointcloud reconstruction. | [GitHub (`facebookresearch/vggt`)](https://github.com/facebookresearch/vggt) • [Local PDF](docs/references/VGGT.pdf) |
