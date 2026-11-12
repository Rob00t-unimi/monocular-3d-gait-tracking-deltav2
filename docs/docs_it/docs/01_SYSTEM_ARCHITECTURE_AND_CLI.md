# 01 — Architettura del Sistema, Gestione dello Stato e CLI Unificata (`gait360`)

## 1. Panoramica Architetturale

`monocular-3d-gait-tracking-deltav2` (**Gait360**) è organizzato in sette directory funzionali principali:

```text
monocular-3d-gait-tracking-deltav2/
├── README.md                          # Panoramica del progetto, guida CLI e sintesi benchmark
├── CHANGELOG.md                       # Cronologia dei rilasci Keep-a-Changelog (v0.1.0, v0.5.0, v1.0.0)
├── pyproject.toml                     # Configurazione del package ed entrypoint CLI `gait360`
├── .gitignore                         # Esclusione dataset pesanti, pesi e log temporanei
├── .gitattributes                     # Tracciamento Git LFS per artefatti .npz e report .pdf
├── index.html                         # Hub interattivo HTML e visualizzatore bilingue (GitHub Pages)
├── configs/
│   ├── default_config.yaml            # Iperparametri e percorsi della pipeline
│   ├── envs/                          # Ambienti Conda YAML (delta_env, analysis_env, ecc.)
│   └── setup/                         # Script di setup multipiattaforma e verifica diagnostica CUDA/DELTAv2
├── data/                              # Manifest, stati memorizzati, dataset (Gait_Dataset) e cache IMU (gt_cache)
├── docs/
│   ├── 01_SYSTEM_ARCHITECTURE_AND_CLI.md
│   ├── 02_STEP_BY_STEP_METHODOLOGY.md
│   ├── 03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md
│   ├── 04_MATHEMATICAL_FORMULATION_AND_FIXES.md
│   ├── 05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md
│   ├── 06_DUMMY_BASELINE_METHOD.md
│   ├── CONCLUSIONS.md                     # Sintesi esecutiva: DELTAv2 vs Dummy sui 4 indicatori chiave
│   ├── docs_it/                       # Mirror italiano della documentazione (README, capitoli, report)
│   └── references/                    # PDF degli articoli scientifici di riferimento (DELTAv2, DELTA, SpaTrackerV2, VGGT)
├── examples/                          # 5 Jupyter Notebook interattivi passo-passo (01..05)
├── experiments/                       # Report storici di decisione e prototipi
│   ├── 01_spatracker_vggt/            # Fase 1: Esplorazione bidirezionale VGGT e SpaTrackerV2
│   ├── 02_delta_v1/                   # Fase 2: Tracking denso 3D DELTA v1 e ancore anatomiche
│   └── 03_deltav2_prototyping/        # Fase 3: Validazione DELTAv2 e JSON coorte a 16 video
├── models/                            # Modelli 3D esterni e sottomoduli (DELTA, DELTAv2, SpaTrackerV2, VGGT)
├── results/
│   ├── benchmark_40_metrics.json      # Metriche autorevoli per video e aggregate (coorte 40 video)
│   ├── benchmark_40_summary.csv       # Tabella comparativa sui 4 livelli di valutazione
│   ├── REPORT_40_VIDEOS_BENCHMARK.md  # Report completo benchmark sui 4 livelli (40 video)
│   ├── REPORT_TRIPLET_VS_CLOUD_ABLATION.md
│   └── REPORT_16_VIDEOS_MODEL_COMPARISON.md
├── src/
│   ├── cli.py                         # CLI parametrica unificata (`gait360` / `python -m src.cli`)
│   ├── core/                          # Modelli di dominio, interfacce, costanti ed eccezioni
│   ├── utils/                         # Validatore di percorsi PEP8 e risoluzione radice progetto
│   ├── data_io/                       # I/O video, JSON OpenPose, CSV WitMotion IMU e stati
│   ├── preprocessing/                 # Segmentazione 2D della coscia e filtraggio Butterworth
│   ├── inference/                     # BaseTracker, DummyTracker e DELTAv2Tracker
│   ├── postprocessing/                # Pruner geometrico a 3 stadi ed estrazione tripletta rigida
│   ├── analysis/                      # Allineamento 3D SVD Kabsch ed estrazione segnali cinematici
│   ├── metrics/                       # Metriche di forma d'onda, magnitudo, frequenza e geometria 3D
│   ├── pipeline/                      # Orchestratori di stadio (dataset, coorte, visione, tracking, pruning)
│   ├── benchmark/                     # Valutatore multi-livello e generatore report deterministico
│   └── ui/                            # 4 ispettori grafici interattivi Tkinter (2D, 3D e forme d'onda)
└── tests/                             # 106 test unitari e d'integrazione automatizzati
```

## 2. Caching, Persistenza dello Stato

1. **Cache di Allineamento Trial Multi-Vista (`data/cached_dataset_map.json`):** Serializza la mappatura tra i video RGB (`90°`, `45°`, `Front`, `Rear`), le cartelle dei keypoint 2D OpenPose e i file CSV Ground Truth dei sensori IMU WitMotion per tutti i 59 soggetti.
2. **Cache di Qualità dei Video Candidati (`data/candidate_videos_cache.json`):** Memorizza i punteggi di qualità pre-calcolati sul segnale IMU (`Optimal`, `Good`, `Imprecise`), il Range of Motion (ROM) e la cadenza del passo per consentire una selezione rapida e deterministica della coorte stratificata.
3. **Stato dei Keypoint di Visione 2D (`data/videos_state.json`):** Mantiene persistenti l'indice del frame di riferimento selezionato (`ref_frame`), il frame di inizio camminata (`walk_start`) e le 3 coordinate anatomiche della coscia (`p1`, `p2`, `p3`), con supporto completo alla ripresa automatica.
4. **Cache dei Tensori delle Traiettorie 3D (`data/inference_results/raw/*.npz` e `refined/*.npz`):** Memorizza le nuvole di punti metriche grezze DELTAv2 `(T, N, 3)` e le nuvole filtrate con le triplette rigide `(T, 3, 3)`.

## 3. Interfaccia a Riga di Comando Unificata (`gait360` / `python -m src.cli`)

Tutti gli stadi operativi e gli ispettori visivi sono accessibili attraverso un unico entry-point parametrico:

```bash
# 1. Scansione del dataset e generazione della cache di qualità dei candidati
python -m src.cli scan --mode all

# 2. Selezione deterministica della coorte di 40 video (o apertura GUI con --gui)
python -m src.cli select-cohort

# 3. Segmentazione 2D della coscia ed estrazione dei 3 punti anatomici
python -m src.cli preprocess --config configs/default_config.yaml

# 4. Inferenza delle traiettorie 3D dense (deltav2 o dummy) con ripresa automatica via .npz
python -m src.cli track --tracker deltav2

# 5. Esecuzione del 3-Stage Geometric Pruner (z_weight=0.25, h_min=0.035m, AR<=8.0)
python -m src.cli prune --z-weight 0.25 --min-altitude 0.035 --max-aspect-ratio 8.0

# 6. Calcolo delle metriche cinematiche 3D Kabsch SVD e rigenerazione dei report JSON/CSV/Markdown
python -m src.cli evaluate

# 7. Lancio delle interfacce grafiche interattive (keypoints | cohort | 3d | signals)
python -m src.cli inspect --mode 3d
```
