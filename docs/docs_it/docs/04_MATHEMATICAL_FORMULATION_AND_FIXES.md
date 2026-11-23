# 04 — Formulazione Matematica ed Elaborazione Cinematica

Questo capitolo fornisce la specifica matematica formale della pipeline di post-processing 3D (`src/postprocessing/geometric_pruner.py`), dell'allineamento cinematico camera-mondo (`src/analysis/kinematic_alignment.py`) e delle metriche di disentanglement degli identity swap (`src/metrics/`), allineata con il codice sorgente e il Report Tecnico.

---

## 1. Filtraggio Adattivo a Fase Zero e Pruning Geometrico 2D/3D (`GeometricPruner` v4.1)

Date le traiettorie metriche grezze $\mathbf{X}(t) \in \mathbb{R}^{N_{\text{thigh}} \times 3}$ e le tracce 2D $\mathbf{x}(t) \in \mathbb{R}^{N_{\text{thigh}} \times 2}$ per gli $N_{\text{thigh}} = 203$ punti della coscia (`indici 0..202`):

1. **Pre-Filtraggio Passa-Basso Butterworth a Fase Zero:**
   Tutti i canali 2D e 3D vengono filtrati in avanti e all'indietro (`scipy.signal.filtfilt`, Butterworth del 4° ordine) alla frequenza di taglio fisiologica del dataset $f_c = 2.9296875\text{ Hz}$ stimata tramite profilazione Welch PSD ($f_{\text{low}} = 0.50\text{ Hz}$, $f_{\text{high}} \approx 3 \cdot f_{\text{median}}$).

2. **Pruning per Persistenza dell'Area del Convex Hull 2D:**
   Ad ogni frame $t \in [0, T-1]$, sia $A_{\text{hull}}(t)$ l'area del Convex Hull 2D del cluster attivo e $A_{\text{hull}}(t_{\text{ref}})$ l'area al frame di riferimento. Ogni volta che $A_{\text{hull}}(t) > \gamma_{\max}\,A_{\text{hull}}(t_{\text{ref}})$ (`max_area_ratio = 5.0`), l'outlier periferico più distante viene rimosso iterativamente su **tutti** i frame, garantendo **persistenza temporale al 100%**.

3. **PCA delle Velocità + DBSCAN Adattivo (Isolamento del Cluster Coerente):**
   I profili di velocità vengono standardizzati e proiettati in uno spazio PCA a 5 componenti (`pca_components = 5`), poi raggruppati tramite DBSCAN adattivo a partire da `initial_eps = 0.5` con fattore `1.3` (fino a `15` iterazioni) per isolare il cluster rigido femorale dominante.

---

## 2. Selezione Combinatoria della Tripletta Rigida Ottimale

Dall'insieme degli inlier sopravvissuti $\mathcal{I}$ (sottocampionati al massimo a `80` candidati), le triplette candidate $\tau = (i, j, k) \in \binom{\mathcal{I}}{3}$ vengono pre-filtrate tramite area 3D di riferimento regolarizzata contro gli Z-spike:

$$\tilde{A}_{3\text{D}}(\tau, t_{\text{ref}}) = \frac{A_{3\text{D}}(\tau, t_{\text{ref}})}{1 + w_z \left(\frac{\Delta z_{\max}}{\Delta xy_{\max} + \epsilon}\right)^2}$$

dove $w_z$ (`z_spike_penalty_weight = 2.5` in configurazione) penalizza le triplette gonfiate da picchi di profondità monoculare lungo l'asse ottico. Tra i primi $K = 3000$ candidati pre-filtrati, l'algoritmo esplora soglie adattive di non-collinearità $\rho_{\min} \in \{0.05, 0.02, 0.01, 0.005, 0.001\}$ imponendo $\min_t A_{3\text{D}}(\tau, t) \ge \rho_{\min}\,A_{3\text{D}}^{\text{orig}}(t_{\text{ref}})$ e massimizza lo score composito area-stabilità:

$$\mathcal{S}(\tau) = \ln\!\bigl(1 + A_{2\text{D}}(\tau, t_{\text{ref}})\bigr)\,\ln\!\bigl(1 + \min_t A_{2\text{D}}(\tau, t)\bigr)\cdot \text{Stab}_{2\text{D}}(\tau)\,\text{Stab}_{3\text{D}}(\tau)$$

con $\text{Stab}_{d\text{D}}(\tau) = \mu(A_{d\text{D}}(\tau, \cdot)) / [\sigma(A_{d\text{D}}(\tau, \cdot)) + 10^{-6}]$ per $d \in \{2, 3\}$.

---

## 3. Cinematica in Spazio Camera e Regolarizzazione del Twist sul Piano Tangente

Per una tripletta tracciata $(\mathbf{X}_t^1, \mathbf{X}_t^2, \mathbf{X}_t^3)$ (o una nuvola di $M$ punti):

1. **Accelerazione Lineare del Centroide (Savitzky–Golay):**
   Il centroide 3D istantaneo $\mathbf{c}_t = \frac{1}{M}\sum_{m=1}^{M}\mathbf{X}_t^m$ preserva al 100% le coordinate metriche 3D. L'accelerazione lineare $\mathbf{a}_{\text{cam}}(t)$ è calcolata mediante derivazione seconda di Savitzky–Golay (`window_length = 11`, `polyorder = 2`, `deriv = 2`).

2. **Regolarizzazione del Twist sul Piano Tangente ($\alpha_z = 0.05$) e Gram–Schmidt:**
   Poiché una patch a 3 punti sulla coscia ha un'altezza mediana (braccio di leva) $h = 2.52\text{ cm}$ (IQR $[2.27, 3.27]\text{ cm}$), confrontabile con il jitter relativo di profondità monoculare $\delta z = \sigma(Z_t^m - \bar{Z}_t)$ (mediana $1.37\text{ cm}$), il rumore non regolarizzato induce deflessioni parassite della normale $\Delta\theta_{\mathbf{w}} \approx \arctan(1.37 / 2.52) \approx 28.5^\circ$ (picchi $45^\circ\text{--}90^\circ$). Prima dell'ortogonalizzazione di Gram–Schmidt, attenuamo solo lo scostamento relativo interno attorno alla profondità media $\bar{Z}_t = \frac{1}{3}\sum_{m=1}^{3} Z_t^m$:

   $$\tilde{Z}_t^m = \bar{Z}_t + \alpha_z\,(Z_t^m - \bar{Z}_t), \qquad \alpha_z = 0.05$$

   e costruiamo $\mathbf{R}_{\text{cam}}(t) = [\mathbf{u}_t, \mathbf{v}_t, \mathbf{w}_t] \in \text{SO}(3)$ sui vertici $\tilde{\mathbf{X}}_t^m = (X_t^m, Y_t^m, \tilde{Z}_t^m)^\top$:

   $$\mathbf{u}_t = \frac{\tilde{\mathbf{X}}_t^2 - \tilde{\mathbf{X}}_t^1}{\|\tilde{\mathbf{X}}_t^2 - \tilde{\mathbf{X}}_t^1\|_2}, \qquad \mathbf{w}_t = \frac{\mathbf{u}_t \times (\tilde{\mathbf{X}}_t^3 - \tilde{\mathbf{X}}_t^1)}{\|\mathbf{u}_t \times (\tilde{\mathbf{X}}_t^3 - \tilde{\mathbf{X}}_t^1)\|_2}, \qquad \mathbf{v}_t = \mathbf{w}_t \times \mathbf{u}_t$$

   imponendo continuità di segno $\mathbf{w}_t \cdot \mathbf{w}_{t-1} \ge 0$. Ciò abbatte il jitter angolare mediano a $1.56^\circ$ preservando intatta $\bar{Z}_t$. Per le nuvole di punti (*Refined Cloud SVD*), $\mathbf{w}_t$ ($w_{t,z} \ge 0$) e $\mathbf{y}_t$ sono i vettori singolari destri minore e maggiore della nuvola centrata.

---

## 4. Proiezione IMU nel World Frame, Allineamento Kabsch $\text{SO}(3)$ e Calibrazione $\mathfrak{so}(3)$

1. **World Frame IMU e Rimozione della Gravità Statica:**
   Tramite i quaternioni d'orientamento $\mathbf{R}_{\text{imu}}(t)$, la forza specifica nel frame del sensore viene ruotata nel World frame statico depurando il vettore gravità medio:

   $$\mathbf{a}_{\text{world, IMU}}(t) = \mathbf{R}_{\text{imu}}(t)\,\mathbf{a}_{\text{body}}(t) - \bar{\mathbf{g}}$$

2. **Sincronizzazione Temporale Limitata e Normalizzata sull'Overlap:**
   Il ritardo di clock $\Delta\tau$ tra video RGB e IMU Bluetooth viene stimato massimizzando la cross-correlazione normalizzata sull'overlap entro una finestra limitata $|\tau| \le 2.5\text{ s}$ (`max_lag_seconds = 2.5`).

3. **Rotazione Statica Camera-Mondo ($\mathbf{R}_{\text{cam}\to\text{world}}$):**
   Stimata tramite Kabsch SVD (`scipy.spatial.transform.Rotation.align_vectors`) allineando le accelerazioni visuali 3D filtrate passa-banda $\mathbf{a}_{\text{cam}}(t)$ con $\mathbf{a}_{\text{world, IMU}}(t)$ in $\text{SO}(3)$.

4. **Base Anatomica PCA e Orientamento Relativo in $\mathfrak{so}(3)$:**
   Poiché il piano della tripletta sulla coscia e il piano del sensore IMU appartengono allo stesso corpo rigido ma differiscono per un offset statico incognito, le rotazioni relative $\Delta\mathbf{R}(t) = \mathbf{R}(t_{\text{ref}})^\top \mathbf{R}(t)$ vengono mappate nell'algebra di Lie $\mathfrak{so}(3)$ tramite mappa logaritmica; gli assi anatomici principali sono calibrati tramite PCA (`use_anatomical_pca_basis = True`) per allineare il Pitch sagittale su tutte le viste a 360°.

---

## 5. Disentanglement Biomeccanico degli Identity Swap

Su finestre mobili di $1.2\text{ s}$ (passo $0.2\text{ s}$), un'antifase persistente ($r(t) < -0.35$ per $\ge 35\%$ delle finestre) viene classificata in base alla geometria della camera:
- **Viste con Incrocio Ottico ($90^\circ$ Sagittale e $45^\circ$ Obliquo):** Classificata come **Scambio d'Identità Vero** (*True Identity Swap*, salto del tracker sull'arto controlaterale durante l'occlusione 2D).
- **Viste Assiali ($0^\circ$ Frontale e $180^\circ$ Posteriore):** Gli arti restano separati orizzontalmente dalla larghezza pelvica e non si intersecano mai in 2D; l'antifase riflette uno scorcio prospettico lungo l'asse ottico $Z$ (**Artefatto di Profondità Assiale**).
