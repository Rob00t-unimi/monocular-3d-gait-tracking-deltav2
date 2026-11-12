# 03 — Cronistoria Sperimentale e Selezione del Modello di Tracking 3D

Questo capitolo documenta l'evoluzione cronologica delle architetture valutate durante la fase di ricerca (`experiments/01_spatracker_vggt/`, `experiments/02_delta_v1/`, `experiments/03_deltav2_prototyping/`).

---

## 1. Fase 1: Esplorazione di VGGT e SpaTrackerV2 (`experiments/01_spatracker_vggt/`)

### 1.1 VGGT (Visual Geometry Grounded Transformer)
- **Ipotesi:** Ricostruire nuvole di punti 3D dense della scena direttamente dai frame video monoculari.
- **Modalità di Fallimento sul Cammino:** VGGT assume una scena prevalentemente statica osservata da una telecamera in movimento. Nelle registrazioni di laboratorio clinico del cammino, con una **telecamera fissa** e un **soggetto umano che si muove in modo non rigido**, le pareti di sfondo dominano la geometria globale, mentre il soggetto deambulante subisce uno schiacciamento della profondità (depth flattening) e una grave incoerenza temporale.

### 1.2 SpaTrackerV2 (Tracking Puntuale Bidirezionale Forward-Backward)
- **Ipotesi:** Tracciare i 3 punti anatomici della coscia in avanti e all'indietro a partire dal frame di riferimento ottimale $t_{\text{ref}}$ (selezionato alla massima estensione orizzontale del passo tra le ginocchia).
- **Riscontri Sperimentali (Coorte dei 16 Video Comuni):**
  - **Elevato Tasso di Scambio di Identità (Identity Swaps: 31.2%, 5/16 video):** Durante l'incrocio della gamba controlaterale nelle viste laterali a $90^\circ$ e oblique a $45^\circ$, le query puntiformi isolate si agganciano frequentemente alla gamba opposta in oscillazione.
  - **Bassa Fluidità e Rigidità:** Produce un Jerk RMS medio di `98.4 ± 41.2 m/s³` e una Deformazione Rigida 3D del `68.4% ± 24.5%`.
- **Decisione:** Abbandonare il tracking basato esclusivamente su punti sparsi in favore del tracking 3D denso supportato da ancore anatomiche di contesto.

---

## 2. Fase 2: Transizione al Tracking 3D Denso con DELTA v1 (`experiments/02_delta_v1/`)

- **Innovazione:** Introduzione delle **Ancore Anatomiche Corporee** (`200` punti coscia + `300` ancore anti-swap sul bacino + `500` punti di contesto sul tronco), forzando l'attenzione del transformer a distinguere la coscia ipsilaterale dalla gamba opposta.
- **Riscontri Sperimentali (Coorte dei 16 Video Comuni):**
  - Riduzione degli scambi di identità dal `31.2%` al **`12.5%` (2/16 video)** e incremento della correlazione Pearson $r$ del Pitch da `0.142` a `0.218`.
  - **Limite Tecnico:** DELTA v1 presentava un marcato jitter di profondità monoculare ad alta frequenza (`RMS Jerk = 64.2 ± 28.7 m/s³`) e richiedeva la suddivisione a finestra mobile (chunking) per sequenze lunghe (`>250` frame), introducendo discontinuità artificiali alle giunzioni dei blocchi.

---

## 3. Fase 3: Adozione di DELTAv2 + Geometric Pruning a 3 Stadi (`experiments/03_deltav2_prototyping/`)

- **Svolta Metodologica:** L'aggiornamento a **DELTAv2** combinato con la profondità metrica di **UniDepthV2** e il **3-Stage Geometric Pruner** (`z_weight=0.25`, `h_min >= 3.5 cm`, `AR <= 8.0`) ha eliminato gli artefatti di giunzione e stabilizzato la geometria 3D della coscia lungo tutti gli angoli a 360°.

| Architettura (Coorte 16 Video Comuni) | Pitch Pearson $r$ ($\uparrow$) | Pitch nRMSE (%) ($\downarrow$) | AccX Pearson $r$ ($\uparrow$) | 3D Acc DTW ($\downarrow$) | 3D Rigid Def. (%) ($\downarrow$) | RMS Jerk ($\text{m/s}^3$) ($\downarrow$) | Identity Swaps ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dummy Tracker (OpenPose)** | `0.211 ± 0.312` | `32.8 ± 9.4` | `0.214 ± 0.158` | `1.332 ± 0.310` | `15.8 ± 8.9` *(2D)* | `1418.5 ± 560.2` | `5 / 16 (31.2%)` |
| **2. SpaTrackerV2 (Forward-Backward)** | `0.142 ± 0.285` | `36.4 ± 11.2` | `0.168 ± 0.149` | `1.385 ± 0.295` | `68.4 ± 24.5` | `98.4 ± 41.2` | `5 / 16 (31.2%)` |
| **3. DELTA v1 (Dense 3D)** | `0.218 ± 0.294` | `33.1 ± 8.8` | `0.229 ± 0.161` | `1.278 ± 0.194` | `52.1 ± 18.7` | `64.2 ± 28.7` | `2 / 16 (12.5%)` |
| **4. DELTAv2 Raw (Non Filtrato)** | `0.198 ± 0.264` | `30.2 ± 6.5` | `0.251 ± 0.165` | `1.235 ± 0.162` | `108.4 ± 25.1` | `174.6 ± 60.4` | `2 / 16 (12.5%)` |
| **5. DELTAv2 Refined Triplet (Nostro)** | **`0.312 ± 0.334`** | **`29.8 ± 9.1`** | **`0.268 ± 0.159`** | **`1.218 ± 0.168`** | **`27.9 ± 9.6`** | **`35.8 ± 13.9`** | **`1 / 16 (6.2%)`** |
