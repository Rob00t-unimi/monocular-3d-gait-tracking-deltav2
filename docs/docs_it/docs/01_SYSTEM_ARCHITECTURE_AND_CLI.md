# 01 — Architettura del Sistema, Gestione dello Stato e CLI Unificata (`gait360`)

## 1. Panoramica Architetturale

`monocular-3d-gait-tracking-deltav2` (**Gait360**) è organizzato in sette directory funzionali principali:

```text
monocular-3d-gait-tracking-deltav2/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── configs/
│   ├── default_config.yaml              # Iperparametri della pipeline e percorsi
│   ├── envs/                            # Specifiche YAML degli ambienti Conda
│   └── setup/                           # Script di bootstrap ambiente e diagnostica CUDA/DELTAv2
├── data/                                # Manifest del dataset e NTFS Junctions (Gait_Dataset, gt_cache)
├── docs/                                # Documentazione tecnica in 5 capitoli e paper di riferimento
├── examples/                            # 5 Jupyter Notebooks interattivi end-to-end
├── experiments/                         # Decision report storici e prototipi (01..03)
├── models/                              # Tracker 3D esterni tramite NTFS Junctions (DELTA, DELTAv2)
├── results/                             # Risultati ufficiali in JSON, CSV e 3 report di benchmark
├── src/                                 # Pacchetto software di produzione e CLI unificata
│   ├── cli.py                           # Entry-point parametrico unificato (`gait360`)
│   ├── core/                            # Modelli di dominio, interfacce, costanti, eccezioni
│   ├── utils/                           # Validazione percorsi PEP8 e risoluzione root
│   ├── data_io/                         # I/O video, JSON OpenPose, CSV IMU WitMotion e stato
│   ├── preprocessing/                   # Segmentazione 2D coscia e filtraggio segnali Butterworth
│   ├── inference/                       # BaseTracker, DummyTracker e DELTAv2Tracker
│   ├── postprocessing/                  # 3-Stage GeometricPruner ed estrazione tripletta rigida
│   ├── analysis/                        # Allineamento Kabsch SVD 3D ed estrazione segnali cinematici
│   ├── metrics/                         # Metriche di forma d'onda, ampiezza, frequenza e geometria 3D
│   ├── pipeline/                        # Orchestratori dei singoli stadi (dataset, coorte, visione, tracking, pruning)
│   ├── benchmark/                       # Valutatore multi-livello e generatore deterministico dei report
│   └── ui/                              # 4 Interfacce grafiche Tkinter per 2D/3D/forme d'onda
└── tests/                               # Oltre 100 test unitari e di integrazione automatizzati
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
