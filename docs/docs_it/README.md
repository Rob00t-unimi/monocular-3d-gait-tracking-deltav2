# Monocular-3d-gait-tracking-deltav2: Analisi Cinematica del Cammino 3D Monoculare a 360° via DELTAv2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architettura: DELTAv2](https://img.shields.io/badge/Tracker%203D-DELTAv2%20%2B%20UniDepthV2-success.svg)](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
[![Test: 106 Superati](https://img.shields.io/badge/tests-106%20passed-brightgreen.svg)](tests/)

**`Monocular-3d-gait-tracking-deltav2`** è un framework markerless di analisi cinematica 3D del cammino da singola telecamera monoculare, in grado di estrarre la cinematica clinica della coscia (angolo di pitch flesso-estensione sul piano sagittale, accelerazioni lineari 3D, cadenza del passo e stabilità a corpo rigido) su **angolazioni di ripresa arbitrarie a 360°** (`90°` laterale puro, `45°` obliquo anteriore/posteriore, `0°` frontale in avvicinamento e `180°` posteriore in allontanamento), validato direttamente contro i segnali ground-truth di sensori inerziali indossabili sincronizzati **WitMotion IMU** su una coorte di **40 video stratificati** (39 soggetti).

---

<div align="center">
  <h4> <a href="https://rob00t-unimi.github.io/monocular-3d-gait-tracking-deltav2/">Clicca qui per leggere la Documentazione Completa</a> </h4>
</div>

---

## 1. Struttura del Repository

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

## 2. Interfaccia da Riga di Comando Unificata (`gait360`)

L'intera pipeline è controllata da un'unica interfaccia da riga di comando parametrica, accessibile tramite l'eseguibile `gait360` (dopo `pip install -e .`) oppure direttamente via Python (`python -m src.cli <comando> [opzioni]`).

```text
usage: gait360 [-h] {align-dataset,scan,select-cohort,preprocess,track,prune,evaluate,inspect} ...
```

---

### Riferimento Comandi e Parametri

#### 0. `align-dataset` — Verifica Integrità e Riorganizzazione Dataset
Verifica i file locali rispetto a `data/dataset_manifest.json` tramite hashing parziale veloce SHA-256 e riorganizza automaticamente i file grezzi di Gait360 scaricati da LJMU OpenData nella gerarchia canonica (`Participant -> Experiment -> Test -> Angle`).
- **Sintassi:** `python -m src.cli align-dataset [opzioni]`
- **Parametri:**
  - `--manifest <path>` *(default: `data/dataset_manifest.json`)*: Percorso del manifest autorevole con i percorsi canonici e gli hash SHA-256 attesi.
  - `--dir <path>` *(default: `data/Gait_Dataset`)*: Percorso della cartella locale del dataset da verificare e ristrutturare.
  - `--dry-run`: Anteprima a video di corrispondenze esatte, spostamenti necessari, file mancanti ed extra senza alterare i file su disco.
  - `--delete-extras`: Elimina automaticamente i file extra non censiti nel manifest.
- **Esempi:**
  ```bash
  python -m src.cli align-dataset --dry-run      # Anteprima sicura di integrità e spostamenti
  python -m src.cli align-dataset                # Riorganizzazione dei file nella gerarchia canonica
  python -m src.cli align-dataset --delete-extras # Pulizia dei file extra non tracciati
  ```

---

#### 1. `scan` — Mappatura Multi-Modale del Dataset & Cache di Qualità
Esegue la scansione della directory del dataset multi-vista, associa i keypoint 2D OpenPose ai sensori inerziali WitMotion IMU sincronizzati (CSV) e costruisce la cache di qualità dei candidati.
- **Sintassi:** `python -m src.cli scan [opzioni]`
- **Parametri:**
  - `--mode {all, scan, candidates}` *(default: `all`)*:
    - `scan`: scansiona la gerarchia delle cartelle e genera `data/cached_dataset_map.json`.
    - `candidates`: valuta la qualità dei segnali video e genera `data/candidate_videos_cache.json`.
    - `all`: esegue sia la scansione sia la valutazione della qualità dei candidati.
  - `--dataset-dir <path>` *(default: `data/Gait_Dataset`)*: Percorso della cartella contenente il dataset grezzo multi-vista.
  - `--cache-path <path>` *(default: `data/cached_dataset_map.json`)*: Percorso di output per il JSON della mappa del dataset.
  - `--force`: Forza la rigenerazione della cache di qualità anche se `data/candidate_videos_cache.json` esiste già.
- **Esempi:**
  ```bash
  python -m src.cli scan                         # Riusa la cache esistente (<1s)
  python -m src.cli scan --force                 # Ricalcola la cache di qualità su tutte le prove
  python -m src.cli scan --mode scan             # Re-indicizza solo la gerarchia cartelle
  ```

---

#### 2. `select-cohort` — Selezione della Coorte di Benchmark Stratificata a 40 Video
Seleziona in modo deterministico la coorte di benchmark a 40 video bilanciata e stratificata su tutte le 8 prospettive di ripresa (5 video per classe), garantendo l'unicità dei partecipanti (40 soggetti distinti) e il posizionamento valido del sensore inerziale sulla coscia.
- **Sintassi:** `python -m src.cli select-cohort [opzioni]`
- **Parametri:**
  - `--gui`: Avvia l'interfaccia grafica Tkinter interattiva (`InferenceVideoSelectorGUI`) con tab per categoria, ranking di qualità, vincoli di unicità e playback video integrato. Senza questo flag, viene eseguita in modalità CLI headless deterministica.
  - `--force`: Forza la rigenerazione di `selected_inference_videos_40.csv` e `.txt` anche se i manifest sono già presenti.
- **Artefatti Generati:**
  - `data/selected_inference_videos_40.csv`
  - `data/selected_inference_videos_40.txt`
- **Esempi:**
  ```bash
  python -m src.cli select-cohort                # Esportazione deterministica CLI headless
  python -m src.cli select-cohort --gui          # Apre la GUI Tkinter di selezione interattiva
  ```

---

#### 3. `preprocess` — Segmentazione Visiva 2D & Selezione Frame di Riferimento
Esegue il pre-processing di visione 2D: rileva la silhouette del soggetto con MediaPipe, individua il picco massimo di spread orizzontale del ginocchio nella fase di oscillazione terminale ($|x_{\text{RKnee}} - x_{\text{LKnee}}|$) come frame di riferimento ottimale ed estrae il triangolo anatomico di ancoraggio femorale a 3 punti.
- **Sintassi:** `python -m src.cli preprocess [opzioni]`
- **Parametri:**
  - `--config <path>` *(default: `configs/default_config.yaml`)*: Percorso della configurazione YAML della pipeline.
  - `--gui`: Avvia l'interfaccia grafica Matplotlib per ispezionare e riposizionare manualmente i punti di ancoraggio.
  - `--video <name>` *(default: `None`)*: Nome specifico del video da ispezionare in modalità GUI (es. `P27_E2T2_90.mp4`).
  - `--force`: Forza il ricalcolo della cache IMU ground-truth e degli stati delle ancore 2D (`videos_state.json`).
- **Comportamento di Default:** Esecuzione headless automatica con ripresa rapida (`--auto --resume`). Salta automaticamente i video già validati in `data/videos_state.json` in meno di 0.2s.
- **Esempi:**
  ```bash
  python -m src.cli preprocess                   # Esecuzione rapida automatica con resume (<0.2s se in cache)
  python -m src.cli preprocess --gui --video P27_E2T2_90.mp4  # Ispezione punti ancora per singolo video
  ```

---

#### 4. `track` — Inferenza del Tracciamento Punti 3D Denso
Esegue l'inferenza di tracciamento punti 3D denso utilizzando l'architettura selezionata (`deltav2` o `dummy`), memorizzando i tensori delle traiettorie 3D grezze in `data/inference_results/raw/`.
- **Sintassi:** `python -m src.cli track [opzioni]`
- **Parametri:**
  - `--tracker {deltav2, dummy}` *(default: `deltav2`)*:
    - `deltav2`: Pipeline di produzione che combina la stima di profondità metrica UniDepthV2, il tracking ottico coarse-to-fine di DELTAv2 e il mascheramento anatomico della silhouette.
    - `dummy`: Baseline sollevata in pseudo-3D dai keypoint 2D OpenPose (benchmark di ablazione).
  - `--video <name>` *(default: `None`)*: Nome del file video specifico da tracciare. Se omesso, esegue l'inferenza batch su tutti i 40 video di benchmark.
  - `--force`: Riesegue l'inferenza anche se il tensore di traiettorie `.npz` grezzo esiste già in cache.
- **Esempi:**
  ```bash
  python -m src.cli track --tracker deltav2       # Batch su 40 video con ripresa automatica dai file .npz
  python -m src.cli track --tracker deltav2 --video P27_E2T2_90.mp4  # Tracciamento di un singolo video
  python -m src.cli track --tracker dummy         # Esecuzione della baseline dummy
  python -m src.cli track --force                # Forza il ricalcolo sovrascrivendo i file .npz
  ```

---

#### 5. `prune` — Post-Processing Geometrico a 3 Stadi & Selezione Triplette
Applica la pipeline di post-processing geometrico a 3 stadi sulle traiettorie grezze DELTAv2 per filtrare punti spuri di sfondo, smorzare il jitter di profondità monoculare lungo l'asse Z ed estrarre la tripletta rigida ottimale protetta da twist del vettore normale.
- **Sintassi:** `python -m src.cli prune [opzioni]`
- **Parametri:**
  - `--video <name>` *(default: `None`)*: File video specifico da processare. Se omesso, processa tutti i tensori `.npz` grezzi in batch.
  - `--z-weight <float>` *(default: `0.25`)*: Peso anisotropo per l'asse di profondità Z per il clustering DBSCAN spaziale e il filtro di velocità MAD (Stadi 1-2).
  - `--min-altitude <float>` *(default: `0.035`)*: Altitudine fisica minima del triangolo ($h_{\min}$) in metri per prevenire twist con triplette quasi-collineari.
  - `--max-aspect-ratio <float>` *(default: `8.0`)*: Soglia massima del rapporto d'aspetto del triangolo ($	ext{AR} = b / h$).
  - `--force`: Forza il ricalcolo del pruning anche se il tensore `.npz` raffinato esiste già in `data/inference_results/refined/`.
- **Artefatti Generati:** Tensori di traiettoria raffinati in `data/inference_results/refined/`.
- **Esempi:**
  ```bash
  python -m src.cli prune                        # Pruning in batch su tutti i 40 video con guardrail di default
  python -m src.cli prune --video P27_E2T2_90.mp4 # Pruning su singolo video
  python -m src.cli prune --z-weight 0.20 --min-altitude 0.040
  ```

---

#### 6. `evaluate` — Allineamento Cinematico SVD & Generazione Report di Benchmark
Allinea le traiettorie visuali 3D al sistema di coordinate della ground-truth WitMotion IMU mediante decomposizione SVD di Kabsch, estrae le forme d'onda cliniche del cammino (Pitch, accelerazioni 3D lineari), calcola le metriche di benchmark sui 4 livelli (Pearson $r$, Peak XCorr, nRMSE, errore di ROM, DTW, Rigidità, Jerk) e genera i report autorevoli.
- **Sintassi:** `python -m src.cli evaluate [opzioni]`
- **Parametri:**
  - `--reports-only`: Rigenera i report Markdown e il riepilogo CSV direttamente dal file `results/benchmark_40_metrics.json` esistente senza ricalcolare l'allineamento cinematico SVD.
  - `--force`: Forza il ricalcolo completo dell'allineamento cinematico SVD e delle metriche su tutti i 40 video anche se i risultati sono presenti in cache.
- **Artefatti Generati:**
  - `results/benchmark_40_metrics.json`
  - `results/benchmark_40_summary.csv`
  - `results/REPORT_40_VIDEOS_BENCHMARK.md`
  - `results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`
  - `results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`
- **Esempi:**
  ```bash
  python -m src.cli evaluate                     # Calcolo completo dell'allineamento cinematico e benchmark
  python -m src.cli evaluate --reports-only      # Rigenerazione istantanea dei report Markdown (<1s)
  ```

---

#### 7. `inspect` — Suite di Visualizzazione Grafica Interattiva
Avvia ispettori grafici interattivi per la validazione qualitativa dei keypoint, del bilanciamento della coorte, delle nuvole di punti 3D o delle forme d'onda cinematiche.
- **Sintassi:** `python -m src.cli inspect [opzioni]`
- **Parametri:**
  - `--mode {keypoints, cohort, 3d, signals, dashboard}` *(default: `3d`)*:
    - `keypoints`: Apre l'ispettore GUI delle ancore anatomiche 2D e del frame di riferimento.
    - `cohort`: Apre l'interfaccia interattiva Tkinter per la selezione e ispezione della coorte stratificata.
    - `3d`: Apre l'ispettore di traiettorie 3D (scheletro 3D sincronizzato in tempo reale + segnali IMU).
    - `signals` / `dashboard`: Apre la dashboard interattiva di confronto benchmark 3D e 1D (`GaitBenchmarkDashboard`) con toggle delle sorgenti (`Triplet`, `Cluster`, `Raw`).
  - `--video <name>` *(default: `P27_E2T2_90.mp4`)*: Nome del file video specifico per la modalità `keypoints`.
  - `--model {deltav2, spatracker, compare}` *(default: `deltav2`)*: Modello per la modalità `signals`/`dashboard` (`deltav2` per la coorte a 40 video, `spatracker` o `compare` per il confronto a 6 video).
  - `--start <int>` *(default: `0`)*: Indice iniziale del video (base 0) per la modalità `signals`/`dashboard`.
- **Esempi:**
  ```bash
  python -m src.cli inspect --mode 3d            # Visualizzatore 3D della nuvola di punti e scheletro
  python -m src.cli inspect --mode signals       # Grafici delle forme d'onda (Visione vs IMU)
  python -m src.cli inspect --mode cohort        # GUI per la selezione della coorte
  python -m src.cli inspect --mode keypoints     # GUI per la validazione delle ancore 2D
  ```

---

## 3. Sintesi Benchmark su 40 Video

| Metrica (40 Video Stratificati) | 1. Dummy Tracker | 2. DELTAv2 Raw | 3. DELTAv2 Pruned Cloud | 4. DELTAv2 Refined Triplet |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.305` | `0.185 ± 0.258` | `0.276 ± 0.278` | **`0.295 ± 0.329`** *(vince 21/40 vs Cloud)* |
| **Pitch Peak XCorr** ($\uparrow$) | `0.369 ± 0.175` | `0.331 ± 0.156` | `0.391 ± 0.169` | **`0.415 ± 0.195`** |
| **Pitch nRMSE (%)** ($\downarrow$) | `33.2 ± 9.7` | `30.8 ± 6.7` | `34.4 ± 10.6` | **`30.5 ± 9.3`** |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7% ± 88.2%` | `-15.5% ± 66.2%` | `+45.2% ± 112.9%` | **`+4.1% ± 70.8%`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.163` | `0.243 ± 0.168` | **`0.273 ± 0.152`** | `0.255 ± 0.162` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.345 ± 0.328` | `1.242 ± 0.168` | `1.239 ± 0.160` | **`1.230 ± 0.173`** |
| **Deformazione Rigida 3D (%)** ($\downarrow$) | `16.4 ± 9.4` *(2D)* | `110.9 ± 26.8` | `41.4 ± 20.1` | **`28.4 ± 9.9`** *(vince 39/40 = 97.5% vs Cloud)* |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 592.3` | `179.9 ± 62.7` | **`28.9 ± 11.4`** | `36.6 ± 14.3` |
| **Scambi d'Identità Veri ($90^\circ$ e $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `3 / 30 (10.0%)` | **`3 / 30 (10.0%)`** |
| **Artefatti di Profondità Assiale ($0^\circ$ e $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `1 / 10 (10.0%)` | **`1 / 10 (10.0%)`** |

---

---

## 4. Documentazione e Tutorial Interattivi

- **Mirror della Documentazione in Italiano:** Tutti i capitoli tecnici, i report di benchmark e il changelog sono raccolti in italiano nella cartella [`docs/docs_it/`](docs/docs_it/).
- **Capitoli della Documentazione Tecnica (`docs/`):**
  1. [01 — Architettura di Sistema, Gestione Stati e CLI](docs/01_SYSTEM_ARCHITECTURE_AND_CLI.md)
  2. [02 — Metodologia Passo-Passo della Pipeline](docs/02_STEP_BY_STEP_METHODOLOGY.md)
  3. [03 — Cronaca Sperimentale e Modelli di Tracking 3D](docs/03_EXPERIMENTAL_CHRONICLE_AND_MODELS.md)
  4. [04 — Formulazione Matematica e Fix Critici](docs/04_MATHEMATICAL_FORMULATION_AND_FIXES.md)
  5. [05 — Risultati del Benchmark e Studio di Ablazione](docs/05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md)
  6. [06 — Metodo Baseline Dummy e Ablazioni Sintetiche](docs/06_DUMMY_BASELINE_METHOD.md)
- **Jupyter Notebook Interattivi Passo-Passo (`examples/`):**
  1. [`01_dataset_scanning_and_imu_loading.ipynb`](examples/01_dataset_scanning_and_imu_loading.ipynb)
  2. [`02_thigh_segmentation_and_anchor_selection.ipynb`](examples/02_thigh_segmentation_and_anchor_selection.ipynb)
  3. [`03_deltav2_dense_3d_tracking.ipynb`](examples/03_deltav2_dense_3d_tracking.ipynb)
  4. [`04_geometric_pruning_and_rigid_triplet.ipynb`](examples/04_geometric_pruning_and_rigid_triplet.ipynb)
  5. [`05_kinematic_alignment_and_benchmark_metrics.ipynb`](examples/05_kinematic_alignment_and_benchmark_metrics.ipynb)

---

## 5. Dataset, Sottomoduli e Riferimenti della Letteratura

> [!NOTE]
> **Provenienza del Modello DELTAv2 e dei Pesi Pre-addestrati:**  
> DELTAv2 (*DenseTrack3Dv2*) è un prototipo di ricerca d'avanguardia sviluppato da Snap Research (Ngo et al., *"DELTAv2: Dense Efficient Long-range 3D Tracking for Any video"*, arXiv:2508.01170) come evoluzione di DELTA (CVPR 2024). Non essendo distribuito come package formale PyPI/Conda, l'architettura del modello è stata integrata direttamente dal repository sperimentale ufficiale di Snap Research ([`snap-research/DenseTrack3Dv2`](https://github.com/snap-research/DenseTrack3Dv2)), mentre il checkpoint pre-addestrato ufficiale (`densetrack3dv2.pth`, 205.7 MB, generato a fine settembre 2025) è stato scaricato tramite il mirror Google Drive degli autori (`1Qa9YFAjBFIzrrHHWf8NZMln5YkLfw4Qa`) indicato nel loro README. DELTAv2 è stato introdotto nel nostro framework il **20 Ottobre 2026** (Fase 3 degli esperimenti) per superare l'asimmetria temporale e gli identity swap riscontrati con SpaTrackerV2 e DELTA v1.

| Componente / Sottomodulo | Ruolo nel Framework | Articolo Scientifico e Riferimenti | Repository Ufficiale e Risorse |
| :--- | :--- | :--- | :--- |
| **Gait360** | Dataset Multi-Vista e Ground-Truth IMU | Registrazioni video sincronizzate a 360° su soggetti lungo angoli ed elevazioni arbitrarie (`0°`, `45°`, `90°`, `180°`), abbinate a sensori **WitMotion IMU** a 9-DoF. Rilasciato pubblicamente dalla Liverpool John Moores University (LJMU). | [LJMU OpenData (eprint/133)](https://opendata.ljmu.ac.uk/id/eprint/133/) |
| **DELTAv2** | Tracker 3D Denso Primario | Ngo et al., *"DELTAv2: Dense Efficient Long-range 3D Tracking for Any video"*, arXiv:2508.01170. Modello SOTA di tracking e stima di profondità congiunta. | [GitHub (`snap-research/DenseTrack3Dv2`)](https://github.com/snap-research/DenseTrack3Dv2) • [PDF Locale](docs/references/DELTAv2.pdf) |
| **DELTA** | Tracker 3D di Base (Fase 2, v1) | Babiloni et al., *"DELTA: Dense Efficient Long-range 3D Tracking"*, CVPR 2024. Primo modello 3D denso valutato in Fase 2. | [GitHub (`snap-research/DELTA_densetrack3d`)](https://github.com/snap-research/DELTA_densetrack3d) • [PDF Locale](docs/references/DELTA.pdf) |
| **UniDepth / UniDepthV2** | Backbone di Profondità Metrica | Piccinelli et al., *"UniDepth: Universal Monocular Metric Depth Estimation"*, CVPR 2024. Fornisce la deproiezione metrica reale a DELTAv2. | [GitHub (`lpiccinelli-eth/UniDepth`)](https://github.com/lpiccinelli-eth/UniDepth) • [arXiv:2403.18913](https://arxiv.org/abs/2403.18913) |
| **SpaTracker** | Tracker 3D Esplorativo (v1) | Xiao et al., *"Spatial-Temporal Point Tracking"*, 2024. Valutato nel benchmark esplorativo di tracking bidirezionale. | [GitHub (`henry123-boy/SpaTracker`)](https://github.com/henry123-boy/SpaTracker) |
| **SpaTrackerV2** | Tracker 3D Esplorativo (v2) | Xiao et al., *"Spatial-Temporal Point Tracking v2"*, 2024. Estensione per sequenze multi-scenario del cammino. | [GitHub (`henry123-boy/SpaTrackerV2`)](https://github.com/henry123-boy/SpaTrackerV2) • [PDF Locale](docs/references/SpaTrackerv2.pdf) |
| **VGGT** | Tracker Transformer (Fase 1) | Xu et al., *"Visual Geometry Grounded Transformer for Point Tracking"*, 2024. Valutato per la ricostruzione a nuvola di punti. | [GitHub (`facebookresearch/vggt`)](https://github.com/facebookresearch/vggt) • [PDF Locale](docs/references/VGGT.pdf) |
