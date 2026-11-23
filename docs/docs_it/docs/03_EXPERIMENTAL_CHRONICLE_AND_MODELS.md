# 03 — Cronologia Sperimentale e Selezione dei Modelli di Tracking 3D

Questo capitolo documenta l'evoluzione cronologica delle architetture di tracking 3D valutate durante la ricerca (`experiments/01_spatracker_vggt/`, `experiments/02_delta_v1/`, `experiments/03_deltav2_prototyping/`), allineata con il benchmark autorevole su finestra temporale appaiata (`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`) e la Tabella I del Report Tecnico.

---

## 1. Fase 1: Esplorazione di VGGT e SpatialTrackerV2 (`experiments/01_spatracker_vggt/`)

### 1.1 VGGT (Visual Geometry Grounded Transformer)
- **Riferimento:** J. Wang et al., *"VGGT: Visual Geometry Grounded Transformer"*, *CVPR* 2025 (`arXiv:2503.11651`).
- **Ipotesi:** Stimare mappe di punti 3D dense, intrinseci ed estrinseci di camera direttamente da sequenze monoculari del cammino (`VGGT4Track`).
- **Modalità di Fallimento sul Cammino:** VGGT è progettato principalmente per scene statiche con camera in movimento. In laboratorio con **camera statica** e **soggetto non-rigido in movimento**, mantenere le mappe di attenzione multi-frame satura 24 GB di VRAM GPU oltre ~60–90 frame.

### 1.2 SpatialTrackerV2 (Tracking Bidirezionale Forward-Backward)
- **Riferimento:** Y. Xiao et al., *"SpatialTrackerV2: 3D Point Tracking Made Easy"*, *ICCV* 2025 (`arXiv:2507.12462`).
- **Ipotesi:** Tracciare i punti query sulla coscia bidirezionalmente attorno al frame di riferimento $t_{\text{ref}}$ mediante wrapper forward/prefisso-invertito accoppiato a VGGT4Track.
- **Risultati Empirici (6 Video, Finestra Temporale Appaiata 60–90 frame):**
  - Raggiunge una buona correlazione del Pitch su finestre corte (`+0.546 ± 0.250`) e basso RMS Jerk (`30.4 ± 2.0 m/s³`), **ma soffre di forte elasticità non rigida (`Deformazione Rigida 3D = 56.7% ± 13.3%`, quasi 3× peggiore di DELTAv2)** perché il bundle adjustment globale non impone vincoli di rigidità anatomica.
  - **Collo di Bottiglia di Scalabilità:** Non è in grado di elaborare sequenze complete (164–238+ frame) in un singolo passaggio per limiti di VRAM.
- **Decisione:** Scartato come backbone principale di produzione; mantenuto come riferimento di benchmark su finestra breve.

---

## 2. Fase 2: Transizione al Tracking 3D Denso con DELTA v1 (`experiments/02_delta_v1/`)

- **Riferimento:** T. D. Ngo et al., *"DELTA: Dense Efficient Long-range 3D Tracking for Any Video"*, *ICLR* 2025 (`arXiv:2410.24211`).
- **Innovazione:** Sostituite le griglie diffuse con un'armatura anatomica strutturata da 1003 punti (`3` ancore + `200` coscia + `300` bacino + `500` torso) accoppiata alla profondità metrica **UniDepthV2** e a stitching a blocchi da 60 frame (overlap 25 frame) via SVD di Kabsch-Umeyama.
- **Risultati Empirici (6 Video, Finestra Appaiata):**
  - Correlazione competitiva su finestra corta (`Pitch r = +0.535 ± 0.335`, `AccX r = +0.482 ± 0.117`).
  - **Limite:** Lo stitching SVD tra blocchi introduce micro-discontinuità di posizione ai bordi che esplodono alla derivazione seconda (**`RMS Jerk = 476.3 ± 133.8 m/s³`**, 15.5× più rumoroso di DELTAv2) con deformazione ancora elevata (`46.4% ± 31.0%`).

---

## 3. Fase 3: Adozione di DELTAv2 + Pruning Geometrico a 3 Stadi (`experiments/03_deltav2_prototyping/`)

- **Riferimento:** T. D. Ngo et al., *"DELTAv2: Accelerating Dense 3D Point Tracking"*, `arXiv:2602.18432`, 2026; abbinato a **UniDepthV2** (`L. Piccinelli et al., arXiv:2502.20110, 2025`).
- **Svolta:** Il tracking in singolo passaggio sull'intera sequenza (240+ frame senza chunking, `coarse_to_fine_dense=True`, `n_iters=6`) abbinato a **GeometricPruner v4.1** ha eliminato gli artefatti di giunzione tra blocchi e ottenuto la minima deformazione rigida 3D (`19.6%` su finestra appaiata, `28.4%` sull'intera coorte di 40 video).

### Confronto Multi-Modello su Finestra Temporale Appaiata (6 Prove di Benchmark, 60–90 Frame)

| Architettura / Configurazione | Finestra Temporale | AccX $r$ ($\uparrow$) | Pitch $r$ ($\uparrow$) | Pitch DTW ($\downarrow$) | 3D Acc DTW ($\downarrow$) | Def. Rigida 3D (%) ($\downarrow$) | RMS Jerk ($\text{m/s}^3$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpatialTrackerV2 + VGGT** | Appaiata (60–90 fr) | `+0.360 ± 0.255` | **`+0.546 ± 0.250`** | `0.269 ± 0.075` | `1.167 ± 0.211` | `56.7 ± 13.3` | **`30.4 ± 2.0`** |
| **DELTAv2 (MediaPipe — Matched)** | Appaiata (60–90 fr) | `+0.449 ± 0.255` | `+0.430 ± 0.398` | **`0.221 ± 0.078`** | `1.309 ± 0.077` | **`19.6 ± 7.6`** | `30.7 ± 13.0` |
| **DELTAv2 (BBox Grid — Matched)** | Appaiata (60–90 fr) | **`+0.497 ± 0.218`** | `+0.232 ± 0.298` | `0.312 ± 0.099` | `1.291 ± 0.212` | `24.5 ± 7.8` | `176.7 ± 46.7` |
| **DELTA v1 (Chunked SVD — Matched)** | Appaiata (60–90 fr) | `+0.482 ± 0.117` | `+0.535 ± 0.335` | `0.235 ± 0.053` | `1.242 ± 0.176` | `46.4 ± 31.0` | `476.3 ± 133.8` |
| **Dummy (2D Lift — Matched)** | Appaiata (60–90 fr) | `+0.394 ± 0.271` | `+0.227 ± 0.181` | `0.278 ± 0.055` | **`1.067 ± 0.138`** | `0.0 ± 0.0` *(2D)* | `1236.6 ± 387.6` |
| **DELTAv2 (MediaPipe — Seq. Intera)** | Intera (164–238 fr) | `+0.401 ± 0.121` | `+0.184 ± 0.217` | `0.234 ± 0.020` | `1.166 ± 0.100` | `26.9 ± 5.6` | `35.6 ± 6.9` |

Le tabelle complete per singolo video sono disponibili in [`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md) e la valutazione completa sui 40 video in [`docs/05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md`](05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md).
