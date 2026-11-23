# Monocular-3d-gait-tracking-deltav2: 360-Degree Monocular 3D Gait Analysis via DELTAv2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architecture: DELTAv2](https://img.shields.io/badge/3D%20Tracker-DELTAv2%20%2B%20UniDepthV2-success.svg)](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
[![Tests: 115 Passed](https://img.shields.io/badge/tests-115%20passed-brightgreen.svg)](tests/)

**`Monocular-3d-gait-tracking-deltav2`** is a markerless monocular 3D gait analysis framework that extracts clinical thigh kinematics (sagittal flexion/extension pitch, 3D linear accelerations, step cadence, and rigid-body stability) across **arbitrary 360° camera viewing angles** (`90°` pure lateral, `45°` front/rear oblique, `0°` frontal approaching, and `180°` posterior receding), validated against synchronized wearable **WitMotion IMU** ground-truth sensors across **40 stratified videos** (40 unique subjects).

---

<div align="center">
  <h3>
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/#report_pdf">📄 Read Technical Report (PDF — GitHub Pages)</a>
  </h3>
  <p>
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/docs/report_en.pdf">Download Report (EN PDF)</a> &nbsp;•&nbsp;
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/docs/report_it.pdf">Scarica Report (IT PDF)</a> &nbsp;•&nbsp;
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/docs/presentation_en.html">🎞️ Presentation Slides (EN)</a> &nbsp;•&nbsp;
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/docs/presentation_it.html">🎞️ Slides (IT)</a> &nbsp;•&nbsp;
    <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/#readme">📚 Full Documentation Hub</a>
  </p>
</div>

---

## 1. Repository Structure

```text
monocular-3d-gait-tracking-deltav2/
├── README.md                          # Project overview, CLI guide & benchmark summary
├── CHANGELOG.md                       # Keep-a-Changelog release history (v0.1.0 .. v1.1.0)
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
│   ├── report_en.pdf                  # Technical Report (English, IEEE two-column, 9 pp.)
│   ├── report_it.pdf                  # Technical Report (Italian, IEEE two-column, 9 pp.)
│   ├── presentation_en.html           # Interactive HTML Presentation Slides (English, v4)
│   ├── presentation_it.html           # Interactive HTML Presentation Slides (Italian, v4)
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
│   └── 03_deltav2_prototyping/        # Phase 3: DELTAv2 validation & benchmark prototypes
├── models/                            # External 3D Trackers & submodules (DELTA, DELTAv2, SpaTrackerV2, VGGT)
├── results/                           # Primary 40-video benchmark (`bbox_grid`, 987 pts)
│   ├── benchmark_40_metrics.json      # Authoritative 40-video per-trial & aggregate metrics
│   ├── benchmark_40_summary.csv       # Tabular summary across all 4 evaluation tiers
│   ├── REPORT_40_VIDEOS_BENCHMARK.md  # Full 40-video 4-tier benchmark report
│   ├── REPORT_TRIPLET_VS_CLOUD_ABLATION.md
│   └── REPORT_16_VIDEOS_MODEL_COMPARISON.md
├── results_mediapipe/                 # 40-video MediaPipe silhouette ablation (`mediapipe`, 1003 pts)
│   └── REPORT_40_VIDEOS_BENCHMARK.md
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
└── tests/                             # 115 automated unit & integration tests
```

## 2. Unified Command-Line Interface (`gait360`)

The entire pipeline is driven by a single parametric command-line interface, accessible either via the installed executable `gait360` (after `pip install -e .`) or directly via Python (`python -m src.cli <command> [options]`).

```text
usage: gait360 [-h] {align-dataset,scan,select-cohort,preprocess,track,prune,evaluate,inspect} ...
```

---

### Command Reference & Parameters

#### 0. `align-dataset` — Dataset Integrity Verification & Restructuring
Verifies local files against `data/dataset_manifest.json` using fast SHA-256 partial-chunk hashing and automatically reorganizes raw Gait360 files downloaded from LJMU OpenData into the canonical hierarchy (`Participant -> Experiment -> Test -> Angle`).
- **Syntax:** `python -m src.cli align-dataset [options]`
- **Parameters:**
  - `--manifest <path>` *(default: `data/dataset_manifest.json`)*: Path to the authoritative manifest containing canonical file paths and SHA-256 hashes.
  - `--dir <path>` *(default: `data/Gait_Dataset`)*: Path to the local dataset directory to verify and restructure.
  - `--dry-run`: Preview identified matches, relocations, missing files, and extra files without applying any disk modifications.
  - `--delete-extras`: Automatically delete untracked extra files not present in the manifest.
- **Examples:**
  ```bash
  python -m src.cli align-dataset --dry-run      # Preview verification and proposed moves safely
  python -m src.cli align-dataset                # Restructure files into canonical hierarchy
  python -m src.cli align-dataset --delete-extras # Clean up untracked extra files
  ```

---

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
Performs 2D vision preprocessing: identifies the terminal swing maximum horizontal knee spread peak ($|x_{\text{RKnee}} - x_{\text{LKnee}}|$) as the optimal reference frame and extracts the 3-point anatomical femur anchor triangle via Sobel edge detection and Huber fitting. (Note: MediaPipe silhouette sampling belongs to Stage 3 `DeltaV2Tracker` context sampling, which fell back to a 784-point `28×28` bounding-box grid in the 40-video v1.0.0 run; see `results/REPORT_40_VIDEOS_BENCHMARK.md`.)
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
Executes dense 3D point tracking inference using the chosen tracker architecture (`deltav2` or `dummy`), storing raw 3D trajectory tensors in `data/inference_results/raw/` (for `bbox_grid`) or `data/inference_results/raw_mediapipe/` (for `mediapipe`).
- **Syntax:** `python -m src.cli track [options]`
- **Parameters:**
  - `--tracker {deltav2, dummy}` *(default: `deltav2`)*:
    - `deltav2`: Production pipeline combining UniDepthV2 metric depth estimation with DELTAv2 coarse-to-fine optical tracking and configurable context anchors.
    - `dummy`: 2D OpenPose pseudo-3D lifted baseline (ablation benchmark).
  - `--context-mode {mediapipe, bbox_grid}` *(default: `mediapipe`)*: Context sampling strategy for DELTAv2 (`bbox_grid` = `987` pts routed to `raw/`; `mediapipe` = `1003` pts routed to `raw_mediapipe/`).
  - `--allow-fallback`: Allow explicit fallback to `bbox_grid` if MediaPipe silhouette sampling fails.
  - `--video <name>` *(default: `None`)*: Specific video filename to track. If omitted, runs batch inference across all 40 benchmark videos.
  - `--force`: Re-run inference even if cached raw `.npz` trajectory tensor already exists.
- **Examples:**
  ```bash
  python -m src.cli track --tracker deltav2 --context-mode bbox_grid  # Primary 987-pt batch run
  python -m src.cli track --tracker deltav2 --context-mode mediapipe  # 1003-pt MediaPipe silhouette run
  python -m src.cli track --tracker deltav2 --video P27_E2T2_90.mp4   # Track single video
  python -m src.cli track --tracker dummy                             # Run dummy baseline tracker
  ```

---

#### 5. `prune` — 3-Stage Geometric Post-Processing & Triplet Selection
Executes the 3-Stage Geometric Post-Processing pipeline on raw DELTAv2 trajectories to filter out background floaters, dampen monocular depth z-jitter, and extract the optimal rigid triplet.
- **Syntax:** `python -m src.cli prune [options]`
- **Parameters:**
  - `--video <name>` *(default: `None`)*: Specific video filename to prune. If omitted, processes all raw `.npz` tensors in batch.
  - `--context-mode {bbox_grid, mediapipe}` *(default: `bbox_grid`)*: Selects which raw/refined directory pair to prune (`raw/ -> refined/` or `raw_mediapipe/ -> refined_mediapipe/`).
  - `--z-weight <float>` *(default: `0.25`)*: Anisotropic depth axis weighting parameter exposed via CLI.
  - `--min-altitude <float>` *(default: `0.035`)*: Minimum physical triangle altitude ($h_{\min}$) in meters.
  - `--max-aspect-ratio <float>` *(default: `8.0`)*: Maximum triangle aspect ratio threshold ($\text{AR} = b / h$).
  - `--force`: Force re-pruning even if cached refined `.npz` already exists.
- **Generated Artifacts:** Refined trajectory tensors in `data/inference_results/refined/` (or `refined_mediapipe/`).
- **Examples:**
  ```bash
  python -m src.cli prune                        # Batch pruning across all 40 videos (`bbox_grid`)
  python -m src.cli prune --context-mode mediapipe
  python -m src.cli prune --video P27_E2T2_90.mp4 # Prune single video
  ```

---

#### 6. `evaluate` — SVD Kinematic Alignment & Benchmark Reporting
Aligns 3D visual trajectories with WitMotion IMU ground-truth coordinate systems using Kabsch SVD rotation and $\mathfrak{so}(3)$ Lie-algebra orientation alignment, extracts clinical gait waveforms (Pitch, 3D accelerations), computes 4-tier benchmark metrics (Pearson $r$, nRMSE, ROM error, DTW, Cadence error, Rigidity, Jerk, Identity Swaps), and generates authoritative reports.
- **Syntax:** `python -m src.cli evaluate [options]`
- **Parameters:**
  - `--context-mode {bbox_grid, mediapipe}` *(default: `bbox_grid`)*: Selects which tracking results to evaluate (`results/` vs `results_mediapipe/`).
  - `--reports-only`: Regenerates markdown reports and summary CSV directly from existing `benchmark_40_metrics.json` without recomputing SVD kinematic alignment.
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

## 3. 40-Video Benchmark Summary (`bbox_grid`, N = 40)

| Metric (40 Stratified Videos) | 1. Dummy Tracker (2D Lift) | 2. DELTAv2 Raw | 3. DELTAv2 Refined Cloud (SVD) | 4. DELTAv2 Refined Triplet |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.452` | `0.185 ± 0.316` | `0.276 ± 0.288` | **`0.295 ± 0.329`** *(wins 21/40 vs Cloud)* |
| **Pitch DTW (norm)** ($\downarrow$) | **`0.218 ± 0.057`** | `0.244 ± 0.049` | `0.240 ± 0.044` | `0.243 ± 0.042` *(wins 22/40 vs Cloud)* |
| **Pitch nRMSE (%)** ($\downarrow$) | **`28.6 ± 7.8`** | `30.4 ± 6.7` | `34.4 ± 8.3` | `30.5 ± 9.3` |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7 ± 46.0` | `-15.5 ± 48.5` | `+44.1 ± 54.6` | **`+4.1 ± 70.8`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.140` | `0.243 ± 0.171` | **`0.288 ± 0.145`** | `0.255 ± 0.162` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.244 ± 0.230` | `1.275 ± 0.214` | `1.248 ± 0.170` | **`1.230 ± 0.173`** |
| **3D Rigid Deformation (%)** ($\downarrow$) | `0.0 ± 0.0` *(planar 2D)* | `33.3 ± 12.3` | `41.4 ± 8.3` | **`28.4 ± 9.9`** *(wins 39/40 = 97.5% vs Cloud)* |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 646.6` | `179.9 ± 58.6` | **`36.2 ± 14.1`** | `36.6 ± 14.3` |
| **Antiphase Duration (%)** ($\downarrow$) | `22.3 ± 26.4` | `17.3 ± 17.8` | `16.8 ± 19.9` | **`13.5 ± 15.9`** |
| **True Identity Swaps ($90^\circ$ & $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `5 / 30 (16.7%)` | **`3 / 30 (10.0%)`** |
| **Axial Depth Artifacts ($0^\circ$ & $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `2 / 10 (20.0%)` | **`1 / 10 (10.0%)`** |

### Context-Mode Ablation (`bbox_grid` vs. `mediapipe` — 40/40 Videos)

| Configuration (40 Videos) | Pitch Pearson $r$ ($\uparrow$) | AccX Pearson $r$ ($\uparrow$) | 3D Rigid Def (%) ($\downarrow$) | Pitch ROM Error (%) | True Swaps ($90^\circ$/$45^\circ$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bbox_grid` — DELTAv2 Raw** | `0.185 ± 0.316` | `0.243 ± 0.171` | `33.3 ± 12.3` | `-15.5 ± 48.5` | `4 / 30 (13.3%)` |
| **`mediapipe` — DELTAv2 Raw** | `0.265 ± 0.344` | **`0.289 ± 0.152`** | `35.8 ± 18.9` | `+5.1 ± 65.8` | `6 / 30 (20.0%)` |
| **`bbox_grid` — Refined Cloud (SVD)** | `0.276 ± 0.288` | `0.288 ± 0.145` | `41.4 ± 8.3` | `+44.1 ± 54.6` | `5 / 30 (16.7%)` |
| **`mediapipe` — Refined Cloud (SVD)** | `0.293 ± 0.295` | `0.268 ± 0.134` | `45.1 ± 13.7` | `+36.6 ± 46.2` | `6 / 30 (20.0%)` |
| **`mediapipe` — Refined Triplet** | `0.209 ± 0.385` | `0.258 ± 0.160` | `34.3 ± 18.5` | `+9.9 ± 60.8` | `8 / 30 (26.7%)` |
| **`bbox_grid` — Refined Triplet (Best Overall)** | **`0.295 ± 0.329`** | `0.255 ± 0.162` | **`28.4 ± 9.9`** | **`+4.1 ± 70.8`** | **`3 / 30 (10.0%)`** |

> **Verdict:** Pure `1003`-point MediaPipe silhouette sampling improves `DELTAv2 Raw` (`Pitch r`: `0.185 -> 0.265`, `AccX r`: `0.243 -> 0.289`) and `Refined Cloud (SVD)` (`Pitch r`: `0.276 -> 0.293`), **but `bbox_grid` + `Refined Triplet` remains the best method overall** (`Pitch r = 0.295`, `3/30 = 10.0%` swaps, `28.4%` deformation, `+4.1%` ROM error) because background grid points anchor global scene geometry during sagittal/oblique crossovers. See [`results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md`](results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md).

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
> DELTAv2 (*DenseTrack3Dv2*) is a dense 3D point tracking architecture developed by Ngo et al. (*"DELTAv2: Accelerating Dense 3D Point Tracking"*, arXiv:2602.18432, 2026) as an evolution of DELTA (ICLR 2025). It is not distributed as a package via PyPI. The model architecture was integrated directly from Snap Research's official repository ([`snap-research/DenseTrack3Dv2`](https://github.com/snap-research/DenseTrack3Dv2)), and the official pre-trained checkpoint (`densetrack3dv2.pth`, 205.7 MB) was obtained via the authors' Google Drive mirror documented in their repository to overcome the temporal asymmetry and identity swaps observed with SpatialTrackerV2 and DELTA v1.

| Component / Submodule | Role in Framework | Scientific Paper & References | Upstream Code & Resources |
| :--- | :--- | :--- | :--- |
| **Gait360** | Multi-View Dataset & Wearable IMU Ground-Truth | L. K. Topham et al., *"A diverse and multi-modal gait dataset of indoor and outdoor walks acquired using multiple cameras and sensors"*, *Scientific Data*, vol. 10, p. 320, 2023. Synchronized 360° gait recordings (`0°`, `45°`, `90°`, `180°`) paired with 9-DoF **WitMotion IMUs** (LJMU). | [LJMU OpenData (eprint/133)](https://opendata.ljmu.ac.uk/id/eprint/133/) |
| **DELTAv2** | Primary Dense 3D Point Tracker | T. D. Ngo et al., *"DELTAv2: Accelerating Dense 3D Point Tracking"*, arXiv:2602.18432, 2026. Coarse-to-fine dense 3D tracking with coupled metric depth. | [GitHub (`snap-research/DenseTrack3Dv2`)](https://github.com/snap-research/DenseTrack3Dv2) • [Local PDF](docs/references/DELTAv2.pdf) |
| **DELTA** | Phase 2 Baseline 3D Tracker (v1) | T. D. Ngo et al., *"DELTA: Dense Efficient Long-range 3D Tracking for Any Video"*, in *Proc. ICLR*, 2025 (arXiv:2410.24211). Initial 3D dense tracker evaluated in Phase 2. | [GitHub (`snap-research/DELTA_densetrack3d`)](https://github.com/snap-research/DELTA_densetrack3d) • [Local PDF](docs/references/DELTA.pdf) |
| **UniDepth / UniDepthV2** | Monocular Metric Depth Backbone | L. Piccinelli et al., *"UniDepth: Universal Monocular Metric Depth Estimation"*, *CVPR* 2024; and *"UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler"*, arXiv:2502.20110, 2025. | [GitHub (`lpiccinelli-eth/UniDepth`)](https://github.com/lpiccinelli-eth/UniDepth) • [arXiv:2502.20110](https://arxiv.org/abs/2502.20110) |
| **SpatialTracker / SpatialTrackerV2** | Phase 1 Exploratory 3D Trackers | Y. Xiao et al., *"SpatialTracker: Tracking Any 2D Pixels in 3D Space"*, *CVPR* 2024; and *"SpatialTrackerV2: 3D Point Tracking Made Easy"*, *ICCV* 2025 (arXiv:2507.12462). | [GitHub (`henry123-boy/SpaTrackerV2`)](https://github.com/henry123-boy/SpaTrackerV2) • [Local PDF](docs/references/SpaTrackerv2.pdf) |
| **VGGT** | Phase 1 Visual Geometry Backbone | J. Wang et al., *"VGGT: Visual Geometry Grounded Transformer"*, *CVPR* 2025 (arXiv:2503.11651). Evaluated for multi-frame 3D pointcloud initialization. | [GitHub (`facebookresearch/vggt`)](https://github.com/facebookresearch/vggt) • [Local PDF](docs/references/VGGT.pdf) |
