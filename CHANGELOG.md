# Changelog

All notable changes to the **monocular-3d-gait-tracking-deltav2** (**Gait360**) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Bilingual Technical Report PDFs (`docs/report_en.pdf`, `docs/report_it.pdf`) & Presentation Slides v4 (`docs/presentation_en.html`, `docs/presentation_it.html`):** Integrated full IEEE two-column technical reports (9 pp., 24 verified references) and interactive v4 slide decks (12 core + 3 bonus slides) directly into the GitHub Pages Documentation Hub (`index.html`), with the Technical Report prioritized first and directly deep-linked from `README.md`.

### Changed
- **Documentation & Notebook Alignment:** Synchronized `README.md`, `docs/01..06`, `docs/CONCLUSIONS.md`, `experiments/03_deltav2_prototyping/README.md`, `results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`, their Italian mirrors (`docs/docs_it/`), and Jupyter notebooks (`examples/03..05`) with the authoritative codebase (`src/`), benchmark CSVs (`results/`, `results_mediapipe/`), verified bibliography, and Technical Report.

---

## [1.1.0] - 2026-11-16

### Added
- **Dual Context-Mode Pipeline & 40/40 MediaPipe Benchmark (`src/inference/delta_v2_tracker.py`, `src/cli.py`, `results_mediapipe/`):** Implemented explicit `--context-mode {bbox_grid,mediapipe}` routing (`raw_mediapipe/`, `refined_mediapipe/`, `results_mediapipe/`), MediaPipe `PoseLandmarker` confidence cascade (`0.5 -> 0.2 -> 0.1 -> 0.05 -> 0.02 -> 0.01 -> 0.0`), and fail-fast context sampling prior to depth prediction, completing all 40/40 videos with pure `1003`-point MediaPipe silhouettes.
- **Context-Mode Comparison Findings:** Within the `mediapipe` run, `Refined Cloud (SVD)` (`Pitch r = 0.293 ± 0.295`) and `Raw` (`Pitch r = 0.265 ± 0.344`, `AccX r = 0.289 ± 0.152`) outperform `mediapipe Refined Triplet` (`Pitch r = 0.209 ± 0.385`, `8/30` swaps) and improve over their `bbox_grid` counterparts (`Raw 0.185`, `Cloud 0.276`). However, **`bbox_grid` + `Refined Triplet` remains the best configuration overall** (`Pitch r = 0.295 ± 0.329`, `3/30 = 10.0%` swaps, `28.4%` 3D deformation, `+4.1%` Pitch ROM error).

---

## [1.0.0] - 2026-11-10

### Added
- **Unified Parametric CLI (`src/cli.py`, `gait360`):** Consolidated all pipeline stages and GUI launchers into eight clean subcommands (`scan`, `select-cohort`, `preprocess`, `track`, `prune`, `evaluate`, `align-dataset`, `inspect`), eliminating standalone script clutter.
- **Dataset Aligner Tool (`src/data_io/dataset_aligner.py`, `gait360 align-dataset`):** Added automated validation and perspective classification of raw video files and 2D OpenPose keypoint JSONs against `dataset_manifest.json`.
- **5-Chapter Technical Documentation (`docs/01_*.md`..`docs/05_*.md`) & 5 End-to-End Jupyter Notebooks (`examples/01_*.ipynb`..`examples/05_*.ipynb`):** Complete step-by-step documentation, Italian mirror (`docs/docs_it/`), and interactive tutorials covering every stage of the pipeline.
- **Executive Benchmark Summary (`docs/CONCLUSIONS.md`):** Comprehensive comparative analysis synthesizing DELTAv2 vs Dummy baseline across 4 core indicators (Cadence, Pitch ROM, Cross-Correlation, and 3D Trajectory Rigidity).
- **Interactive Documentation Hub (`index.html`):** Single-page web documentation application for GitHub Pages with dynamic Markdown rendering, KaTeX mathematical typesetting, and bilingual IT/EN interface.
- **40-Video Stratified SOTA Benchmark (`src/benchmark/`):** Full 4-tier evaluation (`Dummy Tracker`, `DELTAv2 Raw`, `DELTAv2 Pruned Cloud`, `DELTAv2 Refined Triplet`) across 8 viewing perspectives (`90°`, `45°`, `0°`, `180°`) and 16-video matched multi-model comparison (`SpaTrackerV2`, `DELTA v1`, `DELTAv2`).

### Fixed
- **Monocular Depth Z-Spike Attenuation (`src/postprocessing/geometric_pruner.py`):** Introduced anisotropic depth scaling (`z_weight = 0.25`) and a physical velocity MAD floor (`0.015 m/frame`), preventing over-pruning and ray-shooting depth spikes.
- **Narrow-Triangle Normal Twist Prevention (`src/postprocessing/geometric_pruner.py`, `src/analysis/kinematic_alignment.py`):** Added tangent-plane twist regularization ($\alpha_z = 0.05$) and adaptive 3D non-collinearity area floors, reducing Pitch ROM Error from `+44.1%` (Refined Cloud) to **`+4.1%`** (Refined Triplet) and increasing Pitch Pearson $r$ to **`0.295 ± 0.329`**.
- **Anatomical IMU Frame & Kinematic World Alignment (`src/analysis/kinematic_alignment.py`):** Enabled anatomical PCA axis calibration (`use_anatomical_pca_basis = True`), Kabsch $\text{SO}(3)$ alignment, and Lie-algebra $\mathfrak{so}(3)$ relative orientation alignment.
- **Bounded Overlap-Normalized Synchronization (`src/analysis/kinematic_alignment.py`):** Bounded temporal lag search to `max_lag_seconds = 2.5 s` with overlap normalization.

---

## [0.5.0] - 2026-10-24

### Added
- **3-Stage Geometric Post-Processing (`src/postprocessing/geometric_pruner.py`):** Implemented spatial DBSCAN clustering, kinematic velocity/acceleration MAD filtering, and combinatorial rigid triplet selection.
- **3D Kinematic Alignment & Metrics Suite (`src/analysis/`, `src/metrics/`):** Added Kabsch SVD 3D rigid-body alignment, zero-phase Butterworth filtering, and waveform/magnitude/frequency/geometry metrics (`Pearson`, `XCorr`, `nRMSE`, `ROM Error`, `Cadence Error`, `3D DTW`, `Rigidity`, `RMS Jerk`).
- **Interactive Inspection GUIs (`src/ui/`):** Added `KeypointValidatorGUI`, `CohortSelectorGUI`, `Trajectory3DInspector`, and `KinematicSignalInspector`.

---

## [0.1.0] - 2026-10-05

### Added
- **Core Domain & Data I/O Infrastructure (`src/core/`, `src/data_io/`, `src/utils/`):** Implemented domain models, OpenCV video reader, OpenPose 2D JSON parser, WitMotion IMU CSV loader, and dataset scanner.
- **2D Vision Thigh Preprocessing (`src/preprocessing/vision/`):** Implemented directional Sobel edge detection, Huber robust thigh boundary line fitting, optimal reference frame selection, and 3-point anatomical thigh query extraction.
- **Dense 3D Point Tracking Engine (`src/inference/`):** Implemented `BaseTracker`, `DummyTracker`, and `DELTAv2Tracker` with anatomical thigh and body context anchors (`200` thigh + `800` pelvis/torso points) after comparative evaluation against `SpaTrackerV2` and `VGGT`.
