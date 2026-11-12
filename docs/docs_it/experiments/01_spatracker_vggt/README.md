# 🔬 Technical Decision Report: Valutazione Esplorativa di SpatialTrackerV2 + VGGT

- **Branch di Esperimento:** `experiment/spatracker-vggt`
- **Modelli Valutati:** `VGGT4Track` (`Yuxihenry/SpatialTrackerV2_Front`) + `SpatialTrackerV2-Offline` (`Predictor`)
- **Verdetto Decisionale:** **Scartato come Backbone Primario di Produzione** (Mantenuto rigorosamente come baseline di confronto su finestra temporale comune)

---

## 1. Ipotesi Sperimentale e Architettura

All'inizio del progetto, **SpatialTrackerV2** combinato con **VGGT (Visual Geometry Grounded Transformer)** è stato esaminato come architettura congiunta per la stima della posa telecamera e il tracking di punti 3D da video di cammino monoculari non calibrati del dataset **Gait360**:
1. **Inizializzazione Geometrica (`VGGT4Track`):** Stima mappe dense di punti 3D (`points_map`), estrinseci telecamera ($\mathbf{R}_{c2w}, \mathbf{T}_{c2w}$), intrinseci (`intrs`) e incertezza di profondità (`unc_metric`).
2. **Tracking Bidirezionale con Pivot Automatico (`spatrack_test.py`):** Poiché SpatialTrackerV2 opera esclusivamente in avanti nel tempo, è stato sviluppato un wrapper bidirezionale attorno al frame di riferimento validato $t_{\text{ref}}$:
   - Il segmento iniziale $[0, t_{\text{ref}}]$ viene invertito tramite `torch.flip`, tracciato in avanti da $t_{\text{ref}}$, e ribaltato con `np.flip`.
   - Il segmento successivo $[t_{\text{ref}}, T_{\text{end}}]$ viene tracciato normalmente in avanti.
   - Entrambe le mezze traiettorie vengono proiettate in coordinate mondo tramite $\mathbf{X}_{\text{world}}(t) = \mathbf{R}_{c2w}(t)\mathbf{X}_{\text{cam}}(t) + \mathbf{T}_{c2w}(t)$ e congiunte a $t_{\text{ref}}$.

---

## 2. Riscontri Quantitativi (Finestra Rigorosamente Comune: 60–90 Frame)

| Metrica | SpatialTrackerV2 + VGGT (Finestra Comune 60–90 fr) | DELTAv2 + MediaPipe (Finestra Comune 60–90 fr) |
| :--- | :---: | :---: |
| **AccX Pearson $r$** | $+0.360 \pm 0.255$ | **$+0.449 \pm 0.255$** |
| **Pitch Pearson $r$** | **$+0.546 \pm 0.250$** | $+0.430 \pm 0.398$ |
| **Pitch DTW (norm)** | $0.269 \pm 0.075$ | **$0.221 \pm 0.078$** |
| **Deformazione Rigida 3D (%)** | $56.7\% \pm 13.3\%$ | **$19.6\% \pm 7.6\%$** |
| **RMS Jerk ($\text{m/s}^3$)** | **$30.4 \pm 2.0$** | $30.7 \pm 13.0$ |

---

## 3. Motivazioni Tecniche per l'Abbandono

1. **Grave Elasticità Spaziale Non Rigida (56.7% di Deformazione):**
   La bundle adjustment globale in SpatialTrackerV2 ottimizza la coerenza di riproiezione 2D senza imporre vincoli di distanza euclidea tra segmenti anatomici. Di conseguenza, il triangolo della coscia tracciato si allunga e si comprime del **56.7% ± 13.3%** (quasi 3 volte peggio rispetto al **19.6% ± 7.6%** di DELTAv2).
2. **Consumo Proibitivo di Memoria GPU (Limite a 60–90 Frame):**
   Mantenere congiuntamente le mappe di attenzione multi-frame di VGGT e i gradienti della bundle adjustment satura 24 GB di VRAM oltre i ~90 frame, rendendo impraticabile l'analisi di una sequenza intera di cammino (180–300 frame, 5–8 cicli di passo) in un singolo passaggio.
3. **Drift Diffuso delle Query di Sfondo:**
   Le query uniformi di sfondo necessarie per stabilizzare la posa telecamera di VGGT si agganciano frequentemente a ombre in movimento o alla gamba controlaterale durante gli incroci sul piano sagittale facendo collassare il tracking.

**Conclusione:** Transizione definitiva a **DELTA / DELTAv2** con stima di profondità metrica (`UniDepthV2`) e mascheramento anatomico delle query.
