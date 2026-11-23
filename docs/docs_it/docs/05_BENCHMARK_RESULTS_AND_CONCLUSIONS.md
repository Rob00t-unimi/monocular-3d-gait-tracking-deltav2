# 05 — Risultati del Benchmark su 40 Video e Studio di Ablazione

Questo capitolo riassume la valutazione quantitativa finale su tutti i **40 video stratificati del cammino** (40 soggetti unici, 8 prospettive di ripresa, 5 video per classe) generata deterministicamente tramite `python -m src.cli evaluate`, allineata con `results/REPORT_40_VIDEOS_BENCHMARK.md`, `results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md` e il Report Tecnico.

---

## 1. Riepilogo Globale sui 4 Livelli (`bbox_grid`, 40 Video)

| Metrica | 1. Dummy Baseline (2D Lift) | 2. DELTAv2 Raw | 3. DELTAv2 Refined Cloud (SVD) | 4. DELTAv2 Refined Triplet (Proposto) |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.452` | `0.185 ± 0.316` | `0.276 ± 0.288` | **`0.295 ± 0.329`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.140` | `0.243 ± 0.171` | **`0.288 ± 0.145`** | `0.255 ± 0.162` |
| **AccY Pearson $r$** ($\uparrow$) | `0.246 ± 0.135` | `0.309 ± 0.185` | **`0.330 ± 0.197`** | `0.324 ± 0.160` |
| **AccZ Pearson $r$** ($\uparrow$) | `0.245 ± 0.150` | **`0.322 ± 0.173`** | `0.275 ± 0.139` | `0.303 ± 0.151` |
| **Pitch DTW (norm)** ($\downarrow$) | **`0.218 ± 0.057`** | `0.244 ± 0.049` | `0.240 ± 0.044` | `0.243 ± 0.042` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.244 ± 0.230` | `1.275 ± 0.214` | `1.248 ± 0.170` | **`1.230 ± 0.173`** |
| **Pitch nRMSE (%)** ($\downarrow$) | **`28.6 ± 7.8`** | `30.4 ± 6.7` | `34.4 ± 8.3` | `30.5 ± 9.3` |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7 ± 46.0` | `-15.5 ± 48.5` | `+44.1 ± 54.6` | **`+4.1 ± 70.8`** |
| **Errore Cadenza (%)** ($\downarrow$) | **`11.7 ± 7.3`** | `25.7 ± 27.4` | `20.1 ± 16.1` | `24.1 ± 26.2` |
| **Deformazione Rigida 3D (%)** ($\downarrow$) | `0.0 ± 0.0` *(planare 2D)* | `33.3 ± 12.3` | `41.4 ± 8.3` | **`28.4 ± 9.9`** |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 646.6` | `179.9 ± 58.6` | **`36.2 ± 14.1`** | `36.6 ± 14.3` |
| **Durata Antifase (%)** ($\downarrow$) | `22.3 ± 26.4` | `17.3 ± 17.8` | `16.8 ± 19.9` | **`13.5 ± 15.9`** |
| **Scambi d'Identità Veri ($90^\circ$ e $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `5 / 30 (16.7%)` | **`3 / 30 (10.0%)`** |
| **Artefatti di Profondità Assiale ($0^\circ$ e $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `2 / 10 (20.0%)` | **`1 / 10 (10.0%)`** |

---

## 2. Ablazione Diretta: Tripletta Rigida Raffinata vs. Nuvola di Punti (SVD)

- **Stabilità a Corpo Rigido 3D (`28.4%` vs `41.4%`):** La Tripletta Raffinata vince su **39 video su 40 (`97.5%`)**, riducendo la deformazione interna non rigida di `13.0` punti percentuali.
- **Forma d'Onda e Ampiezza del Pitch (`r = 0.295` vs `0.276`, Errore ROM `+4.1%` vs `+44.1%`):** La Tripletta Raffinata vince su **21 video su 40** sul Pitch Pearson $r$ (e **22/40** su Pitch DTW) ed elimina la sovrastima ROM del `+44.1%` causata dal movimento periferico dei tessuti molli e degli abiti nella nuvola completa.
- **Smoothing Complementare del Centroide per l'Accelerazione Lineare (`AccX r = 0.288` Cloud vs `0.255` Triplet):** La media spaziale sull'intero cluster persistente funge da filtro passa-basso spaziale naturale contro il rumore di derivazione seconda (`RMS Jerk = 36.2` vs `36.6 m/s³`), mentre la Tripletta Rigida a 3 punti è superiore per cinematica angolare, fedeltà rigida e soppressione degli swap (`3/30` vs `5/30`).

Le tabelle complete per categoria sono disponibili in:
- [`results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- [`results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- [`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- [`results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md`](../../../results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md)

---

## 3. Ablazione del Campionamento di Contesto: `bbox_grid` vs. `mediapipe` (40 Video)

Per isolare l'effetto della distribuzione delle ancore di contesto (griglia `28×28` da `987` punti con sfondo statico vs. silhouette umana MediaPipe pura da `1003` punti), l'intera pipeline sui 40 video è stata rieseguita con `--context-mode mediapipe` (`results_mediapipe/`):

| Configurazione (40 Video) | Pitch Pearson $r$ ($\uparrow$) | AccX Pearson $r$ ($\uparrow$) | Deformazione Rigida 3D (%) ($\downarrow$) | Errore ROM Pitch (%) | Scambi Veri ($90^\circ$/$45^\circ$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bbox_grid` — DELTAv2 Raw** | `0.185 ± 0.316` | `0.243 ± 0.171` | `33.3 ± 12.3` | `-15.5 ± 48.5` | `4 / 30 (13.3%)` |
| **`mediapipe` — DELTAv2 Raw** | `0.265 ± 0.344` | **`0.289 ± 0.152`** | `35.8 ± 18.9` | `+5.1 ± 65.8` | `6 / 30 (20.0%)` |
| **`bbox_grid` — Refined Cloud (SVD)** | `0.276 ± 0.288` | `0.288 ± 0.145` | `41.4 ± 8.3` | `+44.1 ± 54.6` | `5 / 30 (16.7%)` |
| **`mediapipe` — Refined Cloud (SVD)** | `0.293 ± 0.295` | `0.268 ± 0.134` | `45.1 ± 13.7` | `+36.6 ± 46.2` | `6 / 30 (20.0%)` |
| **`mediapipe` — Refined Triplet** | `0.209 ± 0.385` | `0.258 ± 0.160` | `34.3 ± 18.5` | `+9.9 ± 60.8` | `8 / 30 (26.7%)` |
| **`bbox_grid` — Refined Triplet (Migliore in Assoluto)** | **`0.295 ± 0.329`** | `0.255 ± 0.162` | **`28.4 ± 9.9`** | **`+4.1 ± 70.8`** | **`3 / 30 (10.0%)`** |

- **All'interno del run `mediapipe`:** `Refined Cloud (SVD)` (`Pitch r = 0.293`, `6/30` swap) e `Raw` (`Pitch r = 0.265`, `AccX r = 0.289`) superano `mediapipe Refined Triplet` (`Pitch r = 0.209`, `8/30` swap) e migliorano rispetto ai corrispettivi `bbox_grid` `Raw` (`0.185`) e `Cloud` (`0.276`).
- **Vincitore assoluto (`bbox_grid` + `Refined Triplet`):** Tra tutte le combinazioni testate, **`bbox_grid` + `Refined Triplet`** resta la configurazione migliore in assoluto — eguagliando `mediapipe Cloud` sul `Pitch r` (`0.295` vs `0.293`) ma dimezzando gli Scambi d'Identità Veri (`10.0%` vs `20.0%`–`26.7%`) e ottenendo la minima deformazione rigida 3D (`28.4%`) e il minimo bias ROM (`+4.1%`), poiché i punti statici della griglia ancorano la geometria globale durante gli incroci degli arti.
