# 02 — Metodologia Passo-Passo della Pipeline

Questo documento descrive il flusso di lavoro completo in cinque stadi eseguito dai video multi-vista e dai log WitMotion IMU fino alla cinematica 3D validata della coscia, allineato con `src/`, `configs/default_config.yaml` e il Report Tecnico.

---

## Stadio 1: Scansione Dataset, Sincronizzazione e Selezione della Coorte Stratificata

1. **Scansione Soggetti e Prove (`src/data_io/dataset_scanner.py`):** Scansiona `data/Gait_Dataset` (Gait360: 64 partecipanti, 2.942 video a 30 Hz), associando le viste RGB (`90°` laterale, `45°` obliqua, `0°` frontale, `180°` posteriore) ai file JSON OpenPose `BODY_25` e ai CSV WitMotion IMU a 9-DoF (frequenza nominale 83.33 Hz).
2. **Profilazione Spettrale Adattiva Welch PSD e Valutazione IMU (`src/pipeline/dataset_pipeline.py`):** Stima la banda fisiologica del cammino sull'intero dataset (`0.50–2.93 Hz`, highcut di default `2.9296875 Hz`) mediante PSD di Welch, filtra il pitch e le accelerazioni IMU tramite filtro passa-banda Butterworth del 4° ordine a fase zero e classifica ogni prova (`Optimal`, `Good`, `Imprecise`) in base al ROM fisiologico e alla regolarità armonica della cadenza.
3. **Selezione Stratificata Bilanciata su 8 Categorie (`src/pipeline/cohort_selector.py`):** Seleziona deterministicamente **40 video di benchmark** (5 video per classe su 8 categorie direzionali, **40 soggetti unici**, tutti con `sensor_position == "leg"`), salvati in `data/selected_inference_videos_40.csv`.

---

## Stadio 2: Pre-Processing Visivo 2D ed Estrazione delle Ancore Anatomiche sulla Coscia

1. **Selezione del Frame di Riferimento Ottimale (`src/preprocessing/vision/frame_selection.py`):** Valuta i picchi di separazione orizzontale tra le ginocchia (`stride_wave = |RKnee_x - LKnee_x|`, corrispondenti alla massima apertura del passo con occlusione controlaterale minima), la non-intersezione dei segmenti femorali e la confidenza dei keypoint per individuare il frame di riferimento $t_{\text{ref}}$ dopo `walk_start`.
2. **Estrazione Bordi Sobel Direzionali e Regressione Robusta Huber (`src/preprocessing/vision/edge_processing.py`, `line_fitting.py`):** Costruisce la base ortonormale femorale locale $(\mathbf{u}, \mathbf{u}_\perp)$ nella fascia del terzo inferiore della coscia ($0 \le d_\parallel \le 0.35\,L_{\text{femur}}$, $|d_\perp| \le 0.40\,L_{\text{femur}}$), calcola i gradienti direzionali di Sobel (`ksize=3`, sfocatura Gaussiana `7×7`, soglia `15`) e stima i bordi anteriore e posteriore tramite regressione Huber (`max_angle_deg = 20°`).
3. **Erosione Interna di Sicurezza e Tripletta Anatomica 2D (`src/preprocessing/vision/thigh_point_selector.py`):** Erode verso l'interno il poligono segmentato di `2.0 px` (`safety_margin_px = 2.0`) per evitare i salti di profondità sullo sfondo e posiziona un triangolo di ancoraggio 2D ben condizionato $(P_1, P_2, P_3)$ all'interno della coscia, ispezionabile tramite `KeypointValidatorGUI`.

---

## Stadio 3: Tracking Punti 3D Denso con DELTAv2 e Ancore di Contesto

1. **Campionamento Coscia + Contesto Globale (`src/inference/delta_v2_tracker.py`):** Campiona i `3` vertici anatomici più `200` punti densi interni al triangolo femorale (`203` query sulla coscia, indici `0..202`), affiancati da punti di contesto globale in due modalità configurabili:
   - **`bbox_grid` (Benchmark primario, `results/`):** griglia uniforme `28×28 = 784` punti (`987` query totali). I punti di sfondo statici ancorano la geometria globale della scena durante gli incroci sagittali e obliqui.
   - **`mediapipe` (Ablazione silhouette, `results_mediapipe/`):** `800` punti campionati nella silhouette corporea MediaPipe `PoseLandmarker` (`1003` query totali) con cascata di confidenza (`0.5 -> 0.2 -> 0.1 -> 0.05 -> 0.02 -> 0.01 -> 0.0`) e validazione fail-fast prima dell'inferenza di profondità.
2. **Deproiezione 3D Metrica:** Esegue **UniDepthV2** (`lpiccinelli/unidepth-v2-vitl14`, batch size `8`) per la profondità metrica accoppiato a **DELTAv2** (`coarse_to_fine_dense=True`, `window_len=16`, `n_iters=6`), tracciando bidirezionalmente in singolo passaggio e deproiettando le traiettorie in coordinate camera metriche $(X, Y, Z)$ in metri.

---

## Stadio 4: Post-Processing Geometrico a 3 Stadi ed Estrazione della Tripletta Rigida

Implementato in `src/postprocessing/geometric_pruner.py` (`GeometricPruner` v4.1) sulle `203` traiettorie della coscia:
1. **Stadio 0 — Filtraggio Temporale Butterworth a Fase Zero:** Applica un filtro passa-basso Butterworth del 4° ordine a fase zero (`filtfilt`) alla frequenza fisiologica del dataset ($f_c \approx 2.93\text{ Hz}$).
2. **Stadio 1 — Pruning per Persistenza dell'Area del Convex Hull 2D:** Scarta iterativamente i punti di bordo che derivano quando l'area del convex hull 2D del cluster supera `max_area_ratio = 5.0` volte l'area di riferimento, garantendo **persistenza temporale al 100%** su tutti i $T$ frame.
3. **Stadio 2 — PCA delle Velocità + DBSCAN Adattivo (opzionale/ablazione) e Isolamento del Cluster Coerente:** Proietta i profili di velocità 3D in uno spazio PCA a 5 componenti e applica DBSCAN adattivo (`initial_eps = 0.5`, moltiplicatore `1.3`, fino a `15` step) per isolare il cluster femorale rigido dominante.
4. **Stadio 3 — Selezione Combinatoria della Tripletta Rigida Ottimale:** Pre-filtra le triplette candidate tramite area 3D di riferimento penalizzata contro gli Z-spike ($w_z = 2.5$ quando abilitato), impone soglie adattive di non-collinearità 3D (`[0.05, 0.02, 0.01, 0.005, 0.001]` dell'area 3D iniziale) e massimizza lo score composito area-stabilità $\mathcal{S}(\tau)$ (riducendo la Deformazione Rigida 3D a **`28.4% ± 9.9%`**).

---

## Stadio 5: Allineamento Cinematico 3D e Valutazione Multi-Livello

1. **Estrazione Cinematica in Spazio Camera (`src/analysis/kinematic_alignment.py`):** Calcola l'accelerazione lineare 3D del centroide mediante derivazione seconda di **Savitzky-Golay** (`window_length = 11`, `polyorder = 2`, `deriv = 2`). Per le triplette a 3 punti, applica la **regolarizzazione del twist sul piano tangente** ($\alpha_z = 0.05$) attorno alla profondità media istantanea $\bar{Z}_t$ prima dell'ortogonalizzazione di Gram–Schmidt per sopprimere il twist parassita della normale preservando $\bar{Z}_t$; per le nuvole di punti estrae la base best-fit SVD con continuità di segno.
2. **Proiezione IMU nel World Frame, Allineamento Kabsch e Calibrazione $\mathfrak{so}(3)$:** Ruota la forza specifica IMU nel sistema World statico e depura la gravità ($\mathbf{a}_{\text{world}}(t) = \mathbf{R}_{\text{imu}}(t)\mathbf{a}_{\text{body}}(t) - \mathbf{g}$), sincronizza il ritardo temporale tramite cross-correlazione normalizzata sull'overlap limitata a $|\tau| \le 2.5\text{ s}$, stima la rotazione statica $\mathbf{R}_{\text{cam}\to\text{world}} \in \text{SO}(3)$ via Kabsch SVD (`Rotation.align_vectors`), e calibra gli assi IMU non calibrati tramite PCA anatomica (`use_anatomical_pca_basis = True`) e allineamento d'orientamento nell'algebra di Lie $\mathfrak{so}(3)$.
3. **Valutazione Multi-Metrica (`src/benchmark/evaluator.py`):** Valuta 4 livelli (`Dummy`, `DELTAv2 Raw`, `DELTAv2 Refined Cloud`, `DELTAv2 Refined Triplet`) su correlazione di forma d'onda (Pearson $r$), DTW, nRMSE, errore ROM sul Pitch, errore spettrale di cadenza, Deformazione Rigida 3D, RMS Jerk, Scambi d'Identità Veri (`90°`/`45°`) e Artefatti di Profondità Assiale (`0°`/`180°`).
