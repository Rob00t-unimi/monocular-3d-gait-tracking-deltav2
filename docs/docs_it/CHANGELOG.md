# Changelog (Registro delle Modifiche)

Tutte le modifiche rilevanti apportate al progetto **monocular-3d-gait-tracking-deltav2** (**Gait360**) sono documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), e questo progetto segue il [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-10

### Aggiunto
- **CLI Parametrica Unificata (`src/cli.py`, `gait360`):** Consolidamento di tutti gli stadi della pipeline e delle GUI in sette sottocomandi chiari (`scan`, `select-cohort`, `preprocess`, `track`, `prune`, `evaluate`, `inspect`), eliminando la proliferazione di script isolati.
- **Documentazione Tecnica in 5 Capitoli (`docs/01_*.md`..`docs/05_*.md`) e 5 Jupyter Notebooks End-to-End (`examples/01_*.ipynb`..`examples/05_*.ipynb`):** Documentazione metodologica passo-passo completa e tutorial interattivi su ciascuno stadio del framework.
- **Benchmark SOTA Stratificato su 40 Video (`src/benchmark/`):** Valutazione completa a 4 livelli (`Dummy Tracker`, `DELTAv2 Raw`, `DELTAv2 Pruned Cloud`, `DELTAv2 Refined Triplet`) su 8 prospettive di ripresa (`90°`, `45°`, `0°`, `180°`) e confronto multi-modello su 16 video comuni (`SpaTrackerV2`, `DELTA v1`, `DELTAv2`).

### Risolto
- **Attenuazione degli Z-Spike di Profondità Monoculare (`src/postprocessing/geometric_pruner.py`):** Introdotta la pesatura anisotropa della profondità (`z_weight = 0.25`) e una soglia minima fisica (MAD floor `0.015 m/frame`) sulla velocità, prevenendo l'eccesso di pruning e le oscillazioni spurie lungo i raggi ottici.
- **Prevenzione del Twist della Normale nei Triangoli Quasi-Collineari (`src/postprocessing/geometric_pruner.py`):** Introdotti vincoli sull'altezza minima del triangolo anatomico (`h_min >= 0.035 m`) e sull'aspect ratio (`AR <= 8.0`) nello Stadio 3, riducendo l'errore sul ROM di Pitch da `+45.2%` a **`+4.1%`** e aumentando la correlazione Pearson $r$ del Pitch a **`0.295 ± 0.329`**.
- **Terna Anatomica IMU e Rimozione del Leakage da Telecamera (`src/analysis/kinematic_alignment.py`, `src/analysis/signal_extractor.py`):** Disabilitato il rimescolamento non supervisionato degli assi IMU tramite PCA (`use_pca_basis = False`), rimosso il leakage da proiezione sul piano camera SVD globale e introdotta l'integrazione a passaggio singolo della velocità a media nulla.
- **Cross-Correlazione con Segno e Vincolo sul Ritardo Fisiologico (`src/metrics/waveform/cross_correlation.py`):** Sostituita la correlazione non firmata con la cross-correlazione di forma d'onda con segno, vincolata al lag massimo fisiologico di `max_shift_s = 0.5 s`.

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
- **Pre-Processing di Visione 2D della Coscia (`src/preprocessing/vision/`):** Rilevamento contorni Canny, regressione robusta dei bordi della coscia, selezione automatica del frame di riferimento a massima estensione ed estrazione delle 3 ancore anatomiche.
- **Engine di Tracking 3D Denso (`src/inference/`):** Implementazione di `BaseTracker`, `DummyTracker` e `DELTAv2Tracker` con ancore anatomiche di contesto (`200` punti coscia + `800` punti bacino/tronco) a seguito della valutazione comparativa contro `SpaTrackerV2` e `VGGT`.
