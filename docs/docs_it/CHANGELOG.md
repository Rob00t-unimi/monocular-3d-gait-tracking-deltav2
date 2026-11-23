# Changelog (Registro delle Modifiche)

Tutte le modifiche rilevanti apportate al progetto **monocular-3d-gait-tracking-deltav2** (**Gait360**) sono documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), e questo progetto segue il [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Non Rilasciato]

### Aggiunto
- **Report Tecnici PDF Bilingue (`docs/report_it.pdf`, `docs/report_en.pdf`) e Slides v4 (`docs/presentation_it.html`, `docs/presentation_en.html`):** Integrati i report tecnici IEEE a due colonne (9 pp., 24 riferimenti verificati) e le presentazioni interattive v4 (12 slide principali + 3 bonus) nell'Hub GitHub Pages (`index.html`), con il Report Tecnico in prima posizione e link diretto dal `README.md`.

### Modificato
- **Allineamento Documentazione e Notebook:** Sincronizzati `README.md`, `docs/01..06`, `docs/CONCLUSIONS.md`, `experiments/03_deltav2_prototyping/README.md`, `results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`, i mirror italiani (`docs/docs_it/`) e i Jupyter notebook (`examples/03..05`) con codice sorgente (`src/`), CSV di benchmark (`results/`, `results_mediapipe/`), bibliografia verificata e Report Tecnico.

---

## [1.1.0] - 2026-11-16

### Aggiunto
- **Pipeline Dual Context-Mode e Benchmark MediaPipe 40/40 (`src/inference/delta_v2_tracker.py`, `src/cli.py`, `results_mediapipe/`):** Implementato il routing esplicito `--context-mode {bbox_grid,mediapipe}` (`raw_mediapipe/`, `refined_mediapipe/`, `results_mediapipe/`), la cascata di confidenza `PoseLandmarker` (`0.5 -> 0.2 -> 0.1 -> 0.05 -> 0.02 -> 0.01 -> 0.0`) e il campionamento fail-fast prima della stima di profondità, completando 40/40 video con silhouette MediaPipe pure (`1003` punti).
- **Esito del Confronto `bbox_grid` vs `mediapipe`:** Nella run `mediapipe`, `Refined Cloud (SVD)` (`Pitch r = 0.293 ± 0.295`) e `Raw` (`Pitch r = 0.265 ± 0.344`, `AccX r = 0.289 ± 0.152`) superano `mediapipe Refined Triplet` (`Pitch r = 0.209 ± 0.385`, `8/30` scambi) e migliorano rispetto ai rispettivi valori `bbox_grid` (`Raw 0.185`, `Cloud 0.276`). Tuttavia, **`bbox_grid` + `Refined Triplet` resta il metodo migliore in assoluto** (`Pitch r = 0.295 ± 0.329`, `3/30 = 10.0%` scambi, `28.4%` deformazione 3D, `+4.1%` errore ROM Pitch).

---

## [1.0.0] - 2026-11-10

### Aggiunto
- **CLI Parametrica Unificata (`src/cli.py`, `gait360`):** Consolidamento di tutti gli stadi della pipeline e delle GUI in sette sottocomandi chiari (`scan`, `select-cohort`, `preprocess`, `track`, `prune`, `evaluate`, `inspect`), eliminando la proliferazione di script isolati.
- **Documentazione Tecnica in 5 Capitoli (`docs/01_*.md`..`docs/05_*.md`) e 5 Jupyter Notebooks End-to-End (`examples/01_*.ipynb`..`examples/05_*.ipynb`):** Documentazione metodologica passo-passo completa e tutorial interattivi su ciascuno stadio del framework.
- **Benchmark SOTA Stratificato su 40 Video (`src/benchmark/`):** Valutazione completa a 4 livelli (`Dummy Tracker`, `DELTAv2 Raw`, `DELTAv2 Pruned Cloud`, `DELTAv2 Refined Triplet`) su 8 prospettive di ripresa (`90°`, `45°`, `0°`, `180°`) e confronto multi-modello su 16 video comuni (`SpaTrackerV2`, `DELTA v1`, `DELTAv2`).

### Risolto
- **Attenuazione degli Z-Spike di Profondità Monoculare (`src/postprocessing/geometric_pruner.py`):** Introdotta la pesatura anisotropa della profondità (`z_weight = 0.25`) e una soglia minima fisica (MAD floor `0.015 m/frame`) sulla velocità, prevenendo l'eccesso di pruning e le oscillazioni spurie lungo i raggi ottici.
- **Prevenzione del Twist della Normale nei Triangoli Stretti (`src/postprocessing/geometric_pruner.py`, `src/analysis/kinematic_alignment.py`):** Introdotte la regolarizzazione del twist sul piano tangente ($\alpha_z = 0.05$) e soglie adattive di non-collinearità 3D, riducendo l'errore sul ROM di Pitch da `+44.1%` (Refined Cloud) a **`+4.1%`** (Refined Triplet) e portando il Pitch Pearson $r$ a **`0.295 ± 0.329`**.
- **Terna Anatomica IMU e Allineamento Cinematico World (`src/analysis/kinematic_alignment.py`):** Abilitata la calibrazione PCA anatomica degli assi IMU (`use_anatomical_pca_basis = True`), l'allineamento Kabsch in $\text{SO}(3)$ e l'allineamento d'orientamento relativo in $\mathfrak{so}(3)$.
- **Sincronizzazione Limitata Normalizzata sull'Overlap (`src/analysis/kinematic_alignment.py`):** Ricerca del lag temporale vincolata a `max_lag_seconds = 2.5 s` con normalizzazione sull'overlap.

---

## [0.5.0] - 2026-10-24

### Aggiunto
- **Post-Processing Geometrico a 3 Stadi (`src/postprocessing/geometric_pruner.py`):** Implementazione del clustering DBSCAN spaziale, filtraggio cinematico MAD di velocità/accelerazione e selezione combinatoria della tripletta rigida ottimale.
- **Suite di Allineamento Cinematico 3D e Calcolo Metriche (`src/analysis/`, `src/metrics/`):** Aggiunto l'allineamento di corpo rigido 3D Kabsch SVD, filtraggio Butterworth a fase zero e metriche di forma d'onda, ampiezza, frequenza e geometria (`Pearson`, `XCorr`, `nRMSE`, `ROM Error`, `Cadence Error`, `3D DTW`, `Rigidity`, `RMS Jerk`).
- **Interfacce Grafiche di Ispezione Interattiva (`src/ui/`):** Aggiunte `KeypointValidatorGUI`, `CohortSelectorGUI`, `Trajectory3DInspector` e `KinematicSignalInspector`.

---

## [0.1.0] - 2026-10-05

### Aggiunto
- **Infrastruttura Core e I/O Dati (`src/core/`, `src/data_io/`, `src/utils/`):** Modelli di dominio immutabili, lettore video OpenCV, parser JSON 2D OpenPose, loader CSV IMU WitMotion e scanner del dataset multi-vista.
- **Pre-Processing di Visione 2D della Coscia (`src/preprocessing/vision/`):** Rilevamento gradienti direzionali di Sobel, regressione robusta Huber dei bordi della coscia, selezione automatica del frame di riferimento a massima estensione ed estrazione delle 3 ancore anatomiche.
- **Engine di Tracking 3D Denso (`src/inference/`):** Implementazione di `BaseTracker`, `DummyTracker` e `DELTAv2Tracker` con ancore anatomiche di contesto (`200` punti coscia + `800` punti bacino/tronco) a seguito della valutazione comparativa contro `SpaTrackerV2` e `VGGT`.
