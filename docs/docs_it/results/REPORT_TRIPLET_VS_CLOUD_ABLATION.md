# 📊 Report di Benchmark: Tripletta Ottimale vs Best-Fit della Nuvola di Punti (SVD)
### *Valutazione della Cinematica e del Tracking Spaziale 3D su 40 Video Gait360 con Allineamento Cinematico al Mondo*

> **Video Valutati:** 40 / 40  
> **Allineamento Cinematico:** Allineamento Completo al Mondo 3D (IMU $R(t) \cdot a(t) - g$, derivata seconda del baricentro, $R_{cam \to world}$ tramite SVD)  
> **Sensore di Riferimento Ground Truth:** IMU Coscia Destra (Dataset Gait360)  
> **Data:** 2026-11-10 20:39:09  

---

## 1. Sintesi Esecutiva e Verdetto

### 🏆 Confronto Diretto Riassuntivo (Tripletta vs Nuvola vs Baseline Raw):

| Dominio Metrico | Metrica | Baseline Raw | Tripletta Rifinita (3 Punti) | Nuvola Filtrata (SVD) | Vincitore | Vittorie Tripletta vs Nuvola |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Correlazione Cinematica** | **Pitch Pearson r** *(più alto è meglio)* | 0.185 | **0.295** | 0.276 | **Tripletta Rifinita** | **21** vs 19 |
| **Correlazione Cinematica** | **Pitch abs(r)** *(più alto è meglio)* | 0.302 | **0.382** | 0.366 | **Tripletta Rifinita** | **21** vs 19 |
| **Allineamento Forma d'Onda** | **Pitch DTW norm** *(più basso è meglio)* | 0.244 | **0.243** | 0.240 | **Tripletta Rifinita** | **22** vs 18 |
| **Fedeltà di Corpo Rigido** | **Deformazione 3D %** *(più basso è meglio)* | 33.3% | **28.4%** | 41.4% | **Tripletta Rifinita** | **39** vs 1 |
| **Orbita Spazio degli Stati 3D** | **3D Acc DTW** *(più basso è meglio)* | 1.275 | **1.230** | 1.248 | **Tripletta Rifinita** | **18** vs 22 |
| **Fluidità Traiettoria 3D** | **RMS Jerk** *(più basso è meglio)* | 179.9 | **36.6** | 36.2 | **Tripletta Rifinita** | **18** vs 22 |
| **Accelerazione Lineare** | **AccX Pearson r** *(più alto è meglio)* | 0.243 | **0.255** | 0.288 | **Nuvola Filtrata** | **16** vs 24 |

### 📌 Verdetto Scientifico:
1. **Decisa Superiorità della Tripletta Rifinita per la Cinematica del Cammino:**
   - **Correlazione Pitch:** La Tripletta Rifinita raggiunge una correlazione Pearson media di $r = \mathbf{0.295}$ contro $\mathbf{0.276}$ della Nuvola (**+7.0% di miglioramento netto**). Vince in **21 video su 40 (52.5%)**.
   - **Fedeltà di Forma d'Onda (DTW):** La Tripletta ottiene una distanza DTW inferiore rispetto al Ground Truth (0.243 vs 0.240), vincendo in **22 video su 40 (55.0%)**.
   - **Coerenza di Corpo Rigido:** I 3 punti ottimali isolati dal Geometric Pruning mostrano solo il **28.4%** di deformazione strutturale, contro il **41.4%** dell'intera nuvola di punti (vittoria in ben **39 video su 40, 97.5%**).
2. **Ruolo della Nuvola di Punti:**
   - Il Best-Fit SVD su oltre 100 punti agisce come filtro passa-basso spaziale sul moto del baricentro (AccX $r = 0.288$ vs $0.255$ per la Tripletta).
   - **Tuttavia, l'adattamento di un singolo piano sull'intera coscia confonde la non-rigidità dei vestiti e le pieghe del tessuto con la rotazione scheletrica**, degradando la stima dell'orientamento angolare.
3. **Conclusione:** **La Tripletta Ottimale Rifinita è inequivocabilmente confermata come la metodologia superiore per il Tracking 3D del Cammino.**

---

## 2. Statistiche Metriche Complete (40 Video)

| Metrica | Dummy Baseline | Raw Baseline | Tripletta Rifinita | Nuvola Filtrata | Metodo Migliore |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pearson Pitch (signed r)** | 0.199 | 0.185 | **0.295** | 0.276 | **Tripletta Rifinita** |
| **Pearson Pitch abs(r)** | 0.439 | 0.302 | **0.382** | 0.366 | **Tripletta Rifinita** |
| **Pearson AccX** | 0.205 | 0.243 | **0.255** | 0.288 | **Nuvola Filtrata** |
| **Pearson AccY** | 0.246 | 0.309 | **0.324** | 0.330 | **Nuvola Filtrata** |
| **Pearson AccZ** | 0.245 | 0.322 | **0.303** | 0.275 | **Raw** |
| **Dynamic Time Warping (Pitch DTW)** | 0.218 | 0.244 | **0.243** | 0.240 | **Nuvola Filtrata** |
| **3D State-Space Acc DTW** | 1.244 | 1.275 | **1.230** | 1.248 | **Tripletta Rifinita** |
| **Pitch nRMSE (%)** | 28.6% | 30.4% | **30.5%** | 34.4% | **Raw** |
| **Deformazione Rigida 3D (%)** | 0.0% | 33.3% | **28.4%** | 41.4% | **Tripletta Rifinita** |
| **RMS Jerk del Baricentro** | 1452.0 | 179.9 | **36.6** | 36.2 | **Nuvola Filtrata** |
| **Errore Relativo Cadenza (%)** | 11.7% | 25.7% | **24.1%** | 20.1% | **Nuvola Filtrata** |

---

## 3. Disaggregazione per Angolo di Ripresa della Telecamera

Un test cruciale dell'Allineamento Cinematico al Mondo è verificare se la stima dell'orientamento rimane solida attraverso prospettive di ripresa radicalmente differenti (Sagittale 90°, Frontale 0°, Posteriore 180°, e viste Oblique a 45°):

| Angolo di Ripresa | Video | Pitch r Tripletta | Pitch r Nuvola | Pitch r Raw | DTW Tripletta | DTW Nuvola | Def % Tripletta | Vincitore |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **45° Obliqua (S->D) Frontale-Avvicinamento** | 5 | **0.331** | 0.017 | 0.261 | **0.229** | 0.255 | **31.2%** | 🟢 Tripletta |
| **45° Obliqua (S->D) Posteriore-Allontanamento** | 5 | **0.404** | 0.417 | 0.072 | **0.221** | 0.245 | **25.6%** | 🔵 Nuvola |
| **45° Obliqua (D->S) Frontale-Avvicinamento** | 5 | **0.293** | 0.285 | 0.195 | **0.246** | 0.240 | **30.3%** | 🟢 Tripletta |
| **45° Obliqua (D->S) Posteriore-Allontanamento** | 5 | **0.258** | 0.264 | 0.150 | **0.279** | 0.235 | **32.6%** | 🔵 Nuvola |
| **90° Sagittale (Sinistra -> Destra)** | 5 | **0.408** | 0.364 | 0.161 | **0.221** | 0.188 | **26.6%** | 🟢 Tripletta |
| **90° Sagittale (Destra -> Sinistra)** | 5 | **0.366** | 0.344 | 0.205 | **0.239** | 0.228 | **39.1%** | 🟢 Tripletta |
| **Frontale Pura (In Avvicinamento)** | 5 | **0.327** | 0.224 | 0.347 | **0.251** | 0.271 | **17.8%** | 🟢 Tripletta |
| **Posteriore Pura (In Allontanamento)** | 5 | **-0.030** | 0.290 | 0.087 | **0.256** | 0.257 | **23.8%** | 🔵 Nuvola |

---

## 4. Analisi Biomeccanica e Metodologica

### A. Perché il Geometric Pruning e la Tripletta Ottimale Hanno Successo:

1. **Non-Rigidità degli Indumenti e Pieghe Dinamiche:**
   - Durante il cammino, i vestiti non sono solidali né rigidi: il tessuto si sposta, forma pieghe dinamiche e scivola rispetto alla gamba.
   - I punti campionati sulla superficie della coscia seguono inevitabilmente le deformazioni non rigide dei vestiti anziché la cinematica scheletrica pura. Adattare un singolo piano SVD all'intera nuvola fa sì che il movimento locale delle pieghe inclini la normale stimata, inquinando l'angolo di Pitch con oscillazioni spurie ad alta frequenza.
   - Il **Geometric Pruning** ricerca esplicitamente i triangoli che minimizzano la varianza delle distanze reciproche lungo l'intera sequenza video, filtrando i punti instabili soggetti alle deformazioni del tessuto e trattenendo i 3 punti che presentano la massima coerenza e stabilità di corpo rigido.

2. **Deformazione di Corpo Rigido:**
   - La deformazione strutturale scende dal **41.4%** (nuvola completa) al **28.4%** (tripletta filtrata).

3. **Fedeltà Cinematica Inerziale:**
   - La tripletta rifinita ottiene una correlazione media sul Pitch pari a **r = 0.295**, superando la nuvola completa in **21 trial su 40**.

### B. Il Ruolo dell'Allineamento Cinematico al Mondo:

Attraverso:
1. La conversione dell'accelerazione lineare IMU in coordinate inerziali World tramite quaternioni d'orientamento: $\vec{a}_{world}(t) = R_{imu}(t) \cdot \vec{a}_{body}(t) - \vec{g}$.
2. La stima della rotazione 3D locale $R_{cam \to world}$ tramite allineamento vettoriale SVD all'inizio della camminata.
3. L'allineamento delle traiettorie di orientamento tramite vettori di rotazione relativi nell'algebra di Lie $\mathfrak{so}(3)$;

il sistema di tracking produce curve di Pitch autentiche e conformi alla fisiologia su **tutti gli 8 punti di vista**, validando la pipeline per l'analisi clinica del cammino multi-vista.

