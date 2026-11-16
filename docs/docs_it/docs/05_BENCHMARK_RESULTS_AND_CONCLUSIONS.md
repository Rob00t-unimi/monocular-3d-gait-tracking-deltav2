# 05 — Risultati del Benchmark su 40 Video e Studio di Ablazione

Questo capitolo riassume la valutazione quantitativa finale su tutti i **40 video di cammino stratificati** (8 prospettive di ripresa, 5 video per prospettiva) generata in modo deterministico tramite `gait360 evaluate`.

---

## 1. Sintesi Globale delle Prestazioni sui 4 Livelli di Valutazione (40 Video)

| Metrica | 1. Dummy Tracker | 2. DELTAv2 Raw | 3. DELTAv2 Pruned Cloud | 4. DELTAv2 Refined Triplet |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.305` | `0.185 ± 0.258` | `0.276 ± 0.278` | **`0.295 ± 0.329`** |
| **Pitch Peak XCorr** ($\uparrow$) | `0.369 ± 0.175` | `0.331 ± 0.156` | `0.391 ± 0.169` | **`0.415 ± 0.195`** |
| **Pitch nRMSE (%)** ($\downarrow$) | `33.2 ± 9.7` | `30.8 ± 6.7` | `34.4 ± 10.6` | **`30.5 ± 9.3`** |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7% ± 88.2%` | `-15.5% ± 66.2%` | `+45.2% ± 112.9%` | **`+4.1% ± 70.8%`** |
| **Errore Cadenza Pitch (spm)** ($\downarrow$) | `29.6 ± 27.6` | `31.6 ± 24.5` | `17.9 ± 20.5` | **`16.5 ± 20.6`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.163` | `0.243 ± 0.168` | **`0.273 ± 0.152`** | `0.255 ± 0.162` |
| **AccX Peak XCorr** ($\uparrow$) | `0.354 ± 0.119` | `0.371 ± 0.129` | **`0.383 ± 0.117`** | `0.376 ± 0.123` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.345 ± 0.328` | `1.242 ± 0.168` | `1.239 ± 0.160` | **`1.230 ± 0.173`** |
| **Deformazione Rigida 3D (%)** ($\downarrow$) | `16.4 ± 9.4` *(2D)* | `110.9 ± 26.8` | `41.4 ± 20.1` | **`28.4 ± 9.9`** |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 592.3` | `179.9 ± 62.7` | **`28.9 ± 11.4`** | `36.6 ± 14.3` |
| **True Identity Swaps ($90^\circ$ & $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `3 / 30 (10.0%)` | **`3 / 30 (10.0%)`** |
| **Artefatti di Profondità Assiale ($0^\circ$ & $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `1 / 10 (10.0%)` | **`1 / 10 (10.0%)`** |

---

## 2. Confronto Diretto di Ablazione: Tripletta Rigida vs. Nuvola di Punti Filtrata

- **Stabilità di Corpo Rigido 3D (`28.4%` vs `41.4%`):** La Tripletta Rigida vince su ben **39 video su 40 (`97.5%`)**, riducendo la deformazione non rigida interna di `13.0` punti percentuali netti.
- **Forma d'Onda del Pitch e Ampiezza Sagittale (`r = 0.295` vs `0.276`, Errore ROM `+4.1%` vs `+45.2%`):** La Tripletta vince su **21 video su 40** per la correlazione di Pitch Pearson $r$ ed elimina il sovrastima del `+45.2%` sul ROM provocata dai movimenti dei tessuti molli periferici presenti nella nuvola completa.
- **Smussamento Complementare del Baricentro per l'Accelerazione Lineare (`AccX r = 0.273` Nuvola vs `0.255` Tripletta):** La media spaziale su circa 150 punti nella Nuvola Filtrata agisce come un filtro passa-basso spaziale naturale contro il rumore alle differenze finite di derivata seconda (`RMS Jerk = 28.9` vs `36.6 m/s³`), mentre la Tripletta a 3 punti è nettamente superiore per la cinematica angolare e per la fedeltà del corpo rigido.

Tutte le tabelle dettagliate per singola prospettiva sono consultabili in:
- [`results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- [`results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- [`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- [`results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md`](../../../results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md)

---

## 3. Ablazione Campionamento di Contesto: `bbox_grid` vs `mediapipe` (40 Video)

Per isolare l'effetto della distribuzione delle ancore di contesto (griglia `28×28` da `987` punti incluso sfondo vs silhouette umana pura MediaPipe da `1003` punti), l'intera pipeline sui 40 video è stata rieseguita con `--context-mode mediapipe` (`results_mediapipe/`):

| Configurazione (40 Video) | Pitch Pearson $r$ ($\uparrow$) | AccX Pearson $r$ ($\uparrow$) | Deformazione Rigida 3D (%) ($\downarrow$) | Errore ROM Pitch (%) | Scambi Veri ($90^\circ$/$45^\circ$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bbox_grid` — DELTAv2 Raw** | `0.185 ± 0.316` | `0.243 ± 0.171` | `33.3 ± 12.3` | `-15.5 ± 48.5` | `4 / 30 (13.3%)` |
| **`mediapipe` — DELTAv2 Raw** | `0.265 ± 0.344` | **`0.289 ± 0.152`** | `35.8 ± 18.9` | `+5.1 ± 65.8` | `6 / 30 (20.0%)` |
| **`bbox_grid` — Refined Cloud (SVD)** | `0.276 ± 0.288` | `0.288 ± 0.145` | `41.4 ± 8.3` | `+44.1 ± 54.6` | `5 / 30 (16.7%)` |
| **`mediapipe` — Refined Cloud (SVD)** | `0.293 ± 0.295` | `0.268 ± 0.134` | `45.1 ± 13.7` | `+36.6 ± 46.2` | `6 / 30 (20.0%)` |
| **`mediapipe` — Refined Triplet** | `0.209 ± 0.385` | `0.258 ± 0.160` | `34.3 ± 18.5` | `+9.9 ± 60.8` | `8 / 30 (26.7%)` |
| **`bbox_grid` — Refined Triplet (Migliore in Assoluto)** | **`0.295 ± 0.329`** | `0.255 ± 0.162` | **`28.4 ± 9.9`** | **`+4.1 ± 70.8`** | **`3 / 30 (10.0%)`** |

- **Dentro la run `mediapipe`:** `Refined Cloud (SVD)` (`Pitch r = 0.293`, `6/30` scambi) e `Raw` (`Pitch r = 0.265`, `AccX r = 0.289`) superano nettamente `mediapipe Refined Triplet` (`Pitch r = 0.209`, `8/30` scambi), migliorando rispetto ai rispettivi `Raw` (`0.185`) e `Cloud` (`0.276`) con `bbox_grid`.
- **Vincitore assoluto (`bbox_grid` + `Refined Triplet`):** Tra tutte le combinazioni testate, **`bbox_grid` + `Refined Triplet`** resta il metodo migliore in assoluto: eguaglia `mediapipe Cloud` sul `Pitch r` (`0.295` vs `0.293`), dimezza gli scambi d'identità (`10.0%` vs `20.0%`–`26.7%`) e mantiene la minore deformazione rigida (`28.4%`) e il miglior bias ROM (`+4.1%`), grazie ai punti di sfondo che ancorano la scena globale durante gli incroci.
