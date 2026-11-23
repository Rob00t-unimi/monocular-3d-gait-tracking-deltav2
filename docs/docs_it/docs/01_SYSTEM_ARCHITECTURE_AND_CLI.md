# 01 — Architettura di Sistema, Gestione degli Stati e CLI Unificata (`gait360`)

## 1. Panoramica Architetturale

`monocular-3d-gait-tracking-deltav2` (**Gait360**) è organizzato in directory funzionali modulari che separano dati, configurazioni, documentazione, risultati di benchmark e codice sorgente di produzione:

```text
monocular-3d-gait-tracking-deltav2/
├── README.md                          # Panoramica del progetto, guida CLI e sintesi benchmark
├── CHANGELOG.md                       # Cronologia rilasci Keep-a-Changelog (v0.1.0 .. v1.1.0)
├── pyproject.toml                     # Configurazione del package ed entrypoint CLI `gait360`
├── .gitignore                         # Esclusione dataset pesanti, pesi e log temporanei
├── .gitattributes                     # Tracciamento Git LFS per artefatti .npz e .pdf
├── index.html                         # Hub interattivo HTML e visualizzatore bilingue (GitHub Pages)
├── configs/
│   ├── default_config.yaml            # Iperparametri e percorsi della pipeline
│   ├── envs/                          # Ambienti Conda YAML (delta_env, analysis_env, ecc.)
│   └── setup/                         # Script di setup multipiattaforma e diagnostica CUDA/DELTAv2
├── data/                              # Manifest, stati salvati, dataset (Gait_Dataset) e cache IMU (gt_cache)
├── docs/
│   ├── report_it.pdf                  # Report Tecnico (Italiano, IEEE due colonne, 9 pp.)
│   ├── report_en.pdf                  # Technical Report (Inglese, IEEE due colonne, 9 pp.)
│   ├── presentation_it.html           # Slides Interattive HTML (Italiano, v4)
│   ├── presentation_en.html           # Interactive HTML Slides (Inglese, v4)
│   ├── 01_SYSTEM_ARCHITECTURE_AND_CLI.md
│   ├── 02_STEP_BY_STEP_METHODOLOGY.md
│   ├── 03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md
│   ├── 04_MATHEMATICAL_FORMULATION_AND_FIXES.md
│   ├── 05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md
│   ├── 06_DUMMY_BASELINE_METHOD.md
│   ├── CONCLUSIONS.md                 # Sintesi esecutiva: DELTAv2 vs Dummy sugli indicatori chiave
│   ├── docs_it/                       # Mirror italiano della documentazione (README, capitoli, report)
│   └── references/                    # PDF della letteratura di riferimento (DELTAv2, DELTA, SpaTrackerV2, VGGT)
├── examples/                          # 5 Jupyter Notebook interattivi passo-passo (01..05)
├── experiments/                       # Report storici di decisione e prototipi
│   ├── 01_spatracker_vggt/            # Fase 1: Esplorazione bidirezionale VGGT e SpatialTrackerV2
│   ├── 02_delta_v1/                   # Fase 2: Tracking denso 3D DELTA v1 e ancore anatomiche
│   └── 03_deltav2_prototyping/        # Fase 3: Validazione DELTAv2 e prototipi benchmark
├── models/                            # Modelli 3D esterni e sottomoduli (DELTA, DELTAv2, SpaTrackerV2, VGGT)
├── results/                           # Benchmark primario su 40 video (`bbox_grid`, 987 pt)
│   ├── benchmark_40_metrics.json      # Metriche autorevoli per video e aggregate (40 video)
│   ├── benchmark_40_summary.csv       # Riepilogo tabellare sui 4 livelli di valutazione
│   ├── REPORT_40_VIDEOS_BENCHMARK.md  # Report completo benchmark a 4 livelli (40 video)
│   ├── REPORT_TRIPLET_VS_CLOUD_ABLATION.md
│   └── REPORT_16_VIDEOS_MODEL_COMPARISON.md
├── results_mediapipe/                 # Ablazione MediaPipe su 40 video (`mediapipe`, 1003 pt)
│   └── REPORT_40_VIDEOS_BENCHMARK.md
├── src/
│   ├── cli.py                         # CLI parametrica unificata (`gait360` / `python -m src.cli`)
│   ├── core/                          # Modelli di dominio, interfacce, costanti ed eccezioni
│   ├── utils/                         # Validatore percorsi PEP8 e risoluzione root progetto
│   ├── data_io/                       # I/O video, JSON OpenPose, CSV WitMotion IMU e stati
│   ├── preprocessing/                 # Segmentazione 2D della coscia e filtraggio Butterworth
│   ├── inference/                     # BaseTracker, DummyTracker e DELTAv2Tracker
│   ├── postprocessing/                # Pruner geometrico a 3 stadi ed estrazione tripletta rigida
│   ├── analysis/                      # Allineamento 3D SVD Kabsch ed estrazione segnali cinematici
│   ├── metrics/                       # Metriche di forma d'onda, magnitudo, frequenza e geometria 3D
│   ├── pipeline/                      # Orchestratori di stadio (dataset, coorte, visione, tracking, pruning)
│   ├── benchmark/                     # Valutatore multi-livello e generatore report deterministico
│   └── ui/                            # 4 ispettori grafici interattivi Tkinter (2D, 3D, segnali)
└── tests/                             # 115 test unitari e d'integrazione automatizzati
```

## 2. Caching e Persistenza degli Stati

1. **Cache di Allineamento Multi-Vista (`data/cached_dataset_map.json`):** Serializza la mappatura tra video RGB (`90°`, `45°`, `Front`, `Rear`), directory dei keypoint 2D OpenPose (`BODY_25`) e file CSV dei sensori WitMotion IMU sull'intero dataset Gait360 (64 partecipanti totali, 2.942 video).
2. **Cache di Qualità dei Candidati (`data/candidate_videos_cache.json`):** Memorizza gli score pre-calcolati della ground-truth IMU (`Optimal`, `Good`, `Imprecise`), il Range of Motion (ROM) e la cadenza del passo per la selezione stratificata deterministica.
3. **Stato dei Keypoint 2D (`data/videos_state.json`):** Salva il frame di riferimento selezionato (`ref_frame`), il frame di inizio cammino (`walk_start`) e i 3 punti anatomici di ancoraggio sulla coscia (`p1`, `p2`, `p3`) con ripresa automatica.
4. **Cache dei Tensori di Traiettoria 3D (`data/inference_results/`):** Conserva le nuvole di punti metriche grezze DELTAv2 `(T, N, 3)` (`raw/` per `bbox_grid`, `raw_mediapipe/` per `mediapipe`) e i tensori raffinati dal pruning geometrico + triplette rigide `(T, 3, 3)` (`refined/` e `refined_mediapipe/`).

## 3. Interfaccia da Riga di Comando Unificata (`gait360` / `python -m src.cli`)

Tutti gli stadi della pipeline e gli ispettori interattivi sono accessibili da un unico entrypoint parametrico:

```bash
# 0. Verifica integrità dataset e riorganizzazione file grezzi rispetto al manifest
python -m src.cli align-dataset --dry-run

# 1. Scansione dataset e costruzione cache di qualità candidati
python -m src.cli scan --mode all

# 2. Selezione deterministica della coorte stratificata da 40 video (o GUI con --gui)
python -m src.cli select-cohort

# 3. Segmentazione visiva 2D della coscia ed estrazione ancore a 3 punti
python -m src.cli preprocess --config configs/default_config.yaml

# 4. Tracking 3D denso (`deltav2` o `dummy`; context-mode `bbox_grid` o `mediapipe`)
python -m src.cli track --tracker deltav2 --context-mode bbox_grid
python -m src.cli track --tracker deltav2 --context-mode mediapipe

# 5. Pruning Geometrico a 3 Stadi (`bbox_grid` -> refined/, `mediapipe` -> refined_mediapipe/)
python -m src.cli prune --context-mode bbox_grid
python -m src.cli prune --context-mode mediapipe

# 6. Allineamento cinematico SVD 3D, calcolo metriche e rigenerazione report JSON/CSV/MD
python -m src.cli evaluate --context-mode bbox_grid
python -m src.cli evaluate --context-mode mediapipe

# 7. Avvio GUI interattive (keypoints | cohort | 3d | signals | dashboard)
python -m src.cli inspect --mode 3d
```
