# 02 — Metodologia Passo-Passo della Pipeline

Questo documento descrive dettagliatamente il flusso di lavoro a cinque stadi, dai video multi-vista e registrazioni WitMotion IMU grezzi fino all'estrazione e validazione della cinematica 3D della coscia.

---

## Stadio 1: Scansione Dataset, Sincronizzazione e Selezione Coorte Stratificata

1. **Scansione di Soggetti e Trial (`src/data_io/dataset_scanner.py`):** Esegue la scansione di `data/Gait_Dataset` su 59 soggetti, associando le viste RGB della telecamera (`90°` laterale, `45°` obliqua, `0°` frontale, `180°` posteriore) alle predizioni 2D OpenPose (JSON) e ai log CSV dei sensori IMU WitMotion.
2. **Gradazione di Qualità dell'IMU Ground-Truth (`src/pipeline/dataset_pipeline.py`):** Filtra i segnali IMU di Pitch e accelerazione lineare tramite un filtro Butterworth del 4° ordine a fase zero e classifica ciascun trial candidato (`Optimal`, `Good`, `Imprecise`) in base al ROM fisiologico ($15^\circ$–$65^\circ$) e alla regolarità della cadenza armonica.
3. **Selezione Stratificata Bilanciata su 8 Categorie (`src/pipeline/cohort_selector.py`):** Seleziona 40 video di riferimento (5 video per ciascuna delle 8 prospettive di ripresa, coprendo 39 soggetti unici), salvando il manifesto in `data/selected_inference_videos_40.csv`.

---

## Stadio 2: Pre-Processing di Visione 2D ed Estrazione Ancore Anatomiche della Coscia

1. **Ricerca Automatica del Frame di Riferimento Ottimale (`src/preprocessing/vision/frame_selection.py`):** Analizza i picchi di separazione orizzontale tra le ginocchia (`stride_wave = |RKnee_x - LKnee_x|`, corrispondente alla massima estensione del passo in cui i piedi sono più distanti e l'occlusione da incrocio delle gambe è minima) e la confidenza di rilevamento delle articolazioni per individuare il frame ottimale $t_{\text{ref}}$ successivo a `walk_start`.
2. **Contorni Canny e Fitting Robusto dei Bordi (`src/preprocessing/vision/edge_processing.py`, `line_fitting.py`):** Ritaglia la regione d'interesse (ROI) della coscia, calcola i bordi Canny morfologici ed esegue il fitting lineare dei margini anteriore e posteriore della coscia.
3. **Composizione della Tripletta Anatomica 2D (`src/preprocessing/vision/thigh_point_selector.py`):** Costruisce un triangolo 2D ben condizionato ($P_1$ prossimale-anteriore, $P_2$ distale-anteriore, $P_3$ medio-posteriore) all'interno della silhouette della coscia, validabile o modificabile interattivamente tramite `KeypointValidatorGUI`.

---

## Stadio 3: Tracking 3D Denso con DELTAv2 e Ancore Corporee

1. **Campionamento con Ancore Anatomiche e di Contesto (`src/inference/delta_v2_tracker.py`):** Campiona `3` punti ancora e `200` punti densi all'interno del triangolo coscia insieme ai punti di contesto globale: progettato per `800` punti sulla silhouette MediaPipe (`1003` totali), con fallback a griglia `28×28 = 784` punti (`987` totali). *(Nota: nel benchmark v1.0.0 sui 40 video, tutti i 40 run hanno utilizzato il fallback a griglia da `987` punti per via di un fallback silenzioso su MediaPipe.)*
2. **Riproiezione Metrica 3D:** Esegue la stima di profondità metrica assoluta tramite UniDepthV2 combinata con l'attenzione spazio-temporale del transformer di DELTAv2, proiettando le tracce 2D+profondità in coordinate metriche della telecamera $(X, Y, Z)$ in metri (`data/inference_results/raw/delta_v2_raw_*.npz`).

---

## Stadio 4: Post-Processing Geometrico a 3 Stadi ed Estrazione della Tripletta Rigida

1. **Stadio 1 — DBSCAN Spaziale con Pesatura Anisotropa della Profondità Monoculare (`src/postprocessing/geometric_pruner.py`):** Attenua il rumore di ray-shooting lungo l'asse ottico $Z$ applicando $z_{\text{weight}} = 0.25$ prima di eseguire il clustering 3D DBSCAN frame per frame (`eps=0.12`).
2. **Stadio 2 — Consenso Cinematico MAD su Velocità e Accelerazione:** Calcola i profili istantanei di velocità e accelerazione con $z_{\text{weight}}=0.25$ ed elimina i punti il cui residuo mediano supera $2.5 \times \text{MAD}$ (protetto da una soglia minima fisica di $0.015\text{ m/frame}$).
3. **Stadio 3 — Selezione della Tripletta Rigida Non Degenere:** Valuta tutte le triplette candidate all'interno del cluster sopravvissuto della coscia, imponendo un'altezza minima $h_{\min} \ge 3.5\text{ cm}$ e un aspect ratio massimo $\text{AR} \le 8.0$ per prevenire il twist spurio del vettore normale, minimizzando al contempo la deviazione standard relativa delle lunghezze dei lati (`Deformazione Rigida 3D = 28.4% ± 9.9%`).

---

## Stadio 5: Allineamento Cinematico 3D e Valutazione Multi-Livello del Benchmark

1. **Allineamento di Corpo Rigido 3D Kabsch SVD (`src/analysis/kinematic_alignment.py`):** Allinea il vettore di accelerazione 3D derivato dalla visione (derivata seconda del baricentro meno la gravità) con l'accelerometro 3D del sensore IMU nel sistema anatomico della coscia (`use_pca_basis=False`) tramite rotazione $\text{SO}(3)$ SVD locale e integrazione single-pass della velocità a media nulla.
2. **Valutazione Multi-Metrica (`src/benchmark/evaluator.py`):** Calcola la fedeltà della forma d'onda (Pearson $r$, Peak Cross-Correlation con segno vincolata a un ritardo massimo di $\pm 0.5\text{ s}$), l'accuratezza di ampiezza (nRMSE, errore percentuale sul ROM), l'errore spettrale sulla cadenza del passo e la stabilità geometrica 3D (DTW, Deformazione Rigida, Jerk RMS, Identity Swaps e Artefatti di Profondità Assiali).
