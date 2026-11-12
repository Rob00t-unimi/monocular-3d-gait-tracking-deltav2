# Changelog

All notable changes to the **monocular-3d-gait-tracking-deltav2** (**Gait360**) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Narrow-Triangle Normal Twist Prevention (`src/postprocessing/geometric_pruner.py`):** Added minimum triangle altitude (`h_min >= 0.035 m`) and aspect ratio (`AR <= 8.0`) constraints in Stage 3, reducing Pitch ROM Error from `+45.2%` to **`+4.1%`** and increasing Pitch Pearson $r$ to **`0.295 ± 0.329`**.
- **Anatomical IMU Frame & Camera-Leak Removal (`src/analysis/kinematic_alignment.py`, `src/analysis/signal_extractor.py`):** Disabled unsupervised IMU PCA axis scrambling (`use_pca_basis = False`), removed global SVD camera-plane projection leakage, and enforced single-pass zero-mean velocity integration.
- **Signed Cross-Correlation with Physiological Lag Bound (`src/metrics/waveform/cross_correlation.py`):** Replaced unsigned Euclidean norm correlation with signed waveform cross-correlation capped at `max_shift_s = 0.5 s`.

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
- **2D Vision Thigh Preprocessing (`src/preprocessing/vision/`):** Implemented Canny edge processing, thigh boundary line fitting, optimal reference frame selection, and 3-point anatomical thigh query extraction.
- **Dense 3D Point Tracking Engine (`src/inference/`):** Implemented `BaseTracker`, `DummyTracker`, and `DELTAv2Tracker` with anatomical thigh and body context anchors (`200` thigh + `800` pelvis/torso points) after comparative evaluation against `SpaTrackerV2` and `VGGT`.
