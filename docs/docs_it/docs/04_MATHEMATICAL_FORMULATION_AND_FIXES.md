# 04 — Formulazione Matematica e Trattamento Cinematico

Questo capitolo fornisce la specifica matematica formale del post-processing 3D, dei sistemi di coordinate spaziali, dell'allineamento cinematico di corpo rigido e delle metriche di validazione dei segnali.

---

## 1. Tensore Metrico Anisotropo per la Profondità Monoculare (`z_weight = 0.25`)

Gli stimatori di profondità metrica monoculare (UniDepthV2) presentano un'incertezza significativamente maggiore lungo l'asse ottico $Z$ rispetto agli assi trasversali del piano immagine $(X, Y)$, a causa della geometria proiettiva e dell'ambiguità sui contorni tra soggetto e sfondo (Z-spikes).
Nello **Stadio 1 (Clustering DBSCAN Spaziale)** e nello **Stadio 2 (Pruning Cinematico MAD)** del `GeometricPruner`, le distanze spaziali e le derivate delle traiettorie sono calcolate tramite un tensore metrico anisotropo $\mathbf{W}_z$:

$$\mathbf{W}_z = \text{diag}(1,\; 1,\; w_z), \qquad w_z = 0.25$$

La distanza tra due punti $\mathbf{p}_a, \mathbf{p}_b \in \mathbb{R}^3$ è definita come:

$$d_{\mathbf{W}_z}(\mathbf{p}_a, \mathbf{p}_b) = \|\mathbf{W}_z(\mathbf{p}_a - \mathbf{p}_b)\|_2 = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2 + w_z^2(z_a - z_b)^2}$$

Nello Stadio 2, le deviazioni di velocità rispetto al profilo mediano del cluster vengono filtrate tramite una soglia basata sulla Median Absolute Deviation (MAD), con una soglia minima fisica di rumore $\sigma_{\min} = 0.015\text{ m/frame}$:

$$\tau_v = \max\!\left(2.5 \cdot 1.4826 \cdot \text{MAD}(\|\mathbf{W}_z(\mathbf{v}_i - \bar{\mathbf{v}})\|_2),\; \sigma_{\min}\right)$$

---

## 2. Condizionamento Geometrico delle Triplette di Corpo Rigido (Altezza e Aspect Ratio)

Data una tripletta candidata di punti 3D $(\mathbf{p}_1(t), \mathbf{p}_2(t), \mathbf{p}_3(t)) \in \mathbb{R}^{3 \times 3}$, siano i vettori dei lati $\mathbf{e}_{12} = \mathbf{p}_2 - \mathbf{p}_1$, $\mathbf{e}_{23} = \mathbf{p}_3 - \mathbf{p}_2$ e $\mathbf{e}_{31} = \mathbf{p}_1 - \mathbf{p}_3$, con lunghezza massima $L_{\max}(t)$ e minima $L_{\min}(t)$.
L'area istantanea del triangolo $A(t)$ e la minima altezza fisica (braccio di leva) $h_{\min}(t)$ sono date da:

$$A(t) = \frac{1}{2}\|\mathbf{e}_{12}(t) \times (\mathbf{p}_3(t) - \mathbf{p}_1(t))\|_2, \qquad h_{\min}(t) = \frac{2A(t)}{L_{\max}(t)}$$

Il versore normale alla superficie $\hat{\mathbf{n}}(t)$ è definito come:

$$\hat{\mathbf{n}}(t) = \frac{\mathbf{e}_{12}(t) \times \mathbf{e}_{13}(t)}{\|\mathbf{e}_{12}(t) \times \mathbf{e}_{13}(t)\|_2}$$

Quando $h_{\min}$ è ridotto (triangoli quasi collineari), minime variazioni di profondità $\delta z$ lungo l'asse ottico inducono ampie rotazioni spurie $O(\delta z / h_{\min})$ di $\hat{\mathbf{n}}(t)$. Per garantire la stabilità numerica del vettore normale della coscia, lo Stadio 3 impone due vincoli geometrici valutati sia sul frame di riferimento che sulla media temporale:

$$h_{\min} \ge 0.035\text{ m}\;(3.5\text{ cm}), \qquad \text{AR} = \frac{L_{\max}}{L_{\min}} \le 8.0$$

---

## 3. Sistema di Coordinate Anatomiche della Coscia e Assi Sensore IMU

Per valutare il tracking visivo rispetto ai sensori inerziali indossabili senza distorsioni di coordinate:

1. **Base Anatomica della Coscia:** L'orientamento visivo del femore viene tracciato direttamente nello spazio euclideo 3D attraverso il versore normale $\hat{\mathbf{n}}(t)$ e il versore longitudinale del femore $\hat{\mathbf{u}}_{\text{femur}}(t) = \frac{\mathbf{p}_{\text{knee}}(t) - \mathbf{p}_{\text{hip}}(t)}{\|\mathbf{p}_{\text{knee}}(t) - \mathbf{p}_{\text{hip}}(t)\|_2}$. L'angolo sagittale di flesso-estensione (Pitch) è estratto dalla rotazione attorno all'asse anatomico trasversale.
2. **Preservazione degli Assi Sensore Reali (`use_pca_basis = False`):** Il sensore IMU WitMotion è posizionato sulla superficie anteriore della coscia destra. I suoi assi nominali corrispondono a:
   - Asse $X$: Asse longitudinale della gamba (rivolto verso il basso verso il ginocchio).
   - Asse $Y$: Asse anteroposteriore (ortogonale alla superficie anteriore della coscia).
   - Asse $Z$: Asse mediolaterale trasversale.
   Gli assi fisici nominali del sensore vengono preservati direttamente, garantendo la corrispondenza fisica con il frame anatomico dell'arto.

---

## 4. Orientamento di Corpo Rigido Kabsch $\text{SO}(3)$ e Integrazione della Velocità

1. **Allineamento Rigido $\text{SO}(3)$ Ottimale:** Per determinare la trasformazione spaziale statica tra il sistema di coordinate della telecamera e il sistema inerziale World durante la calibrazione iniziale, il sistema impiega l'algoritmo di Kabsch. Dati i vettori di accelerazione visiva a media nulla $\mathbf{A}_{\text{vis}} \in \mathbb{R}^{T \times 3}$ e i corrispondenti vettori IMU $\mathbf{A}_{\text{imu}} \in \mathbb{R}^{T \times 3}$, la matrice di cross-covarianza $\mathbf{H} = \mathbf{A}_{\text{vis}}^\top \mathbf{A}_{\text{imu}}$ viene decomposta tramite Singular Value Decomposition:

   $$\mathbf{H} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

   La rotazione propria ottimale $\mathbf{R}^* \in \text{SO}(3)$ con $\det(\mathbf{R}^*) = +1$ è:

   $$\mathbf{R}^* = \mathbf{V}\,\text{diag}\!\left(1,\; 1,\; \det(\mathbf{V}\mathbf{U}^\top)\right)\mathbf{U}^\top$$

2. **Integrazione della Velocità a Passaggio Singolo a Media Nulla:** La velocità istantanea 3D $\mathbf{v}(t)$ si ottiene dall'accelerazione lineare $\mathbf{a}(t)$ mediante un singolo passaggio di integrazione trapezoidale cumulativa:

   $$\mathbf{v}(t) = \int_0^t \mathbf{a}(\tau)\,d\tau - \frac{1}{T}\int_0^T \left(\int_0^t \mathbf{a}(\tau)\,d\tau\right)dt$$

   Ciò assicura un profilo di velocità privo di deriva e a media nulla lungo la sequenza di passi analizzata.

---

## 5. Cross-Correlazione con Segno e Vincolo di Ritardo Fisiologico

Il ritardo temporale $\tau^*$ tra la velocità angolare derivata dal video $\omega_{\text{vis}}(t)$ e il segnale giroscopico IMU $\omega_{\text{imu}}(t)$ viene calcolato tramite cross-correlazione normalizzata:

$$\rho(\tau) = \frac{\sum_{t} \left(\omega_{\text{vis}}(t) - \bar{\omega}_{\text{vis}}\right)\left(\omega_{\text{imu}}(t + \tau) - \bar{\omega}_{\text{imu}}\right)}{\sqrt{\sum_t \left(\omega_{\text{vis}}(t) - \bar{\omega}_{\text{vis}}\right)^2 \sum_t \left(\omega_{\text{imu}}(t + \tau) - \bar{\omega}_{\text{imu}}\right)^2}}$$

1. **Preservazione del Segno:** La cross-correlazione viene calcolata sulle forme d'onda firmate (mantenendo valori positivi e negativi). Questo garantisce che eventuali errori di tracking in antifase (come inversioni a $180^\circ$ o scambi controlaterali di gamba) producano correlazioni negative anziché falsi punteggi positivi.
2. **Finestra Temporale Fisiologica:** La ricerca del picco di correlazione è limitata entro una finestra fisiologica rigorosa di $\pm 0.5\text{ s}$ rispetto alla sincronizzazione nominale:

$$\tau^* = \arg\max_{|\tau| \le 0.5\,\text{s}} \rho(\tau)$$
