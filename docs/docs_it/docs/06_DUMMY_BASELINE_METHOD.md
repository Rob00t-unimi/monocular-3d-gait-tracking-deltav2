# 06 — Metodo Baseline Geometrico Dummy (Lift 2D OpenPose)

Questo documento fornisce la specifica tecnica dettagliata del metodo geometrico baseline ("Metodo Dummy") implementato in `src/inference/dummy_tracker.py` e valutato nell'intero benchmark Gait360.

---

## 1. Motivazione e Scopo

Per isolare quantitativamente il contributo reale del tracking 3D metrico profondo (DELTAv2) e del post-processing geometrico (GeometricPruner), il benchmark include una baseline geometrica non basata su deep learning di tracking:
- Invece di seguire le feature ottiche nel tempo tramite meccanismi di cross-attention, il **Metodo Dummy** si basa unicamente sui keypoint scheletrici 2D sparsi estratti da **OpenPose** (`RHip`, `RKnee`, `LHip`, `LKnee`).
- Risponde a una fondamentale domanda scientifica di controllo: *È possibile estrarre una cinematica affidabile della coscia semplicemente proiettando e trasferendo i punti iniziali in base alle articolazioni dello scheletro 2D?*

---

## 2. Formulazione Matematica e Trasferimento dei Punti

### A. Base Anatomica della Coscia al Frame di Riferimento
Al frame di riferimento ottimale $t_{\text{ref}}$, tre punti anatomici di query $P_1, P_2, P_3 \in \mathbb{R}^2$ vengono posizionati all'interno della coscia (tramite `ThighPointSelector` o validazione manuale).
Siano $\mathbf{p}_{\text{knee}} \in \mathbb{R}^2$ le coordinate del ginocchio e $\mathbf{p}_{\text{hip}} \in \mathbb{R}^2$ quelle dell'anca.
Il vettore longitudinale della coscia $\mathbf{v}$ e il suo versore $\hat{\mathbf{u}}$ sono definiti come:

$$\mathbf{v} = \mathbf{p}_{\text{hip}} - \mathbf{p}_{\text{knee}}, \qquad L_{\text{femur}} = \|\mathbf{v}\|_2, \qquad \hat{\mathbf{u}} = \frac{\mathbf{v}}{L_{\text{femur}}}$$

Il versore ortogonale trasversale $\hat{\mathbf{u}}_\perp$ è definito nel piano 2D imponendo $\hat{\mathbf{u}} \cdot \hat{\mathbf{u}}_\perp = 0$:

$$\hat{\mathbf{u}}_\perp = \begin{bmatrix} -\hat{u}_y \\ \hat{u}_x \end{bmatrix}$$

### B. Estrazione delle Coordinate Relative Invarianti di Scala
Ciascun punto della coscia $P_i$ ($i \in \{1, 2, 3\}$) viene parametrizzato in coordinate relative invarianti di scala $(\alpha_i, \beta_i)$ rispetto al ginocchio:

$$\Delta\mathbf{p}_i = P_i - \mathbf{p}_{\text{knee}}$$
$$\alpha_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}}{L_{\text{femur}}}, \qquad \beta_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}_\perp}{L_{\text{femur}}}$$

In questo modo, $\alpha_i$ rappresenta la quota longitudinale normalizzata lungo il femore ($0 = \text{ginocchio}$, $1 = \text{anca}$), mentre $\beta_i$ rappresenta lo scostamento trasversale.

### C. Trasferimento Frame per Frame
Per ogni frame successivo $t \in [0, T-1]$ in cui OpenPose rileva ginocchio $\mathbf{p}_{\text{knee}}(t)$ e anca $\mathbf{p}_{\text{hip}}(t)$:
1. Si ricalcolano la lunghezza istantanea $L_{\text{femur}}(t)$, il versore longitudinale $\hat{\mathbf{u}}(t)$ e il versore ortogonale $\hat{\mathbf{u}}_\perp(t)$.
2. Si ricostruiscono le coordinate assolute dei punti:

$$P_i(t) = \mathbf{p}_{\text{knee}}(t) + L_{\text{femur}}(t) \left( \alpha_i \hat{\mathbf{u}}(t) + \beta_i \hat{\mathbf{u}}_\perp(t) \right)$$

Se in un dato frame il rilevamento è assente o occluso, le posizioni vengono interpolate linearmente.

---

## 3. Sollevamento Proiettivo 3D Semplificato (Naive Pinhole Lifting)

Per proiettare i punti 2D trasferiti nello spazio 3D della telecamera $(X, Y, Z)$ senza un modello di profondità metrica denso:
1. **Profondità da Priore Antropometrico:** Assumendo una lunghezza femorale media $L_{\text{phys}} \approx 0.42\text{ m}$ e la focale $f$, la profondità stimata $\tilde{Z}(t)$ è approssimata da:

   $$\tilde{Z}(t) = \frac{f \cdot L_{\text{phys}}}{L_{\text{femur}}(t)}$$

2. **Inversione Proiettiva Pinhole:**

   $$\mathbf{X}_{\text{cam}, i}(t) = \begin{bmatrix} \frac{P_{i, x}(t) - c_x}{f_x} \cdot \tilde{Z}(t) \\ \frac{P_{i, y}(t) - c_y}{f_y} \cdot \tilde{Z}(t) \\ \tilde{Z}(t) \end{bmatrix}$$

---

## 4. Risultati Sperimentali e Limiti

Valutato sui 40 video di Gait360:
1. **Gravissimo Jitter di Accelerazione (RMS Jerk = 1452.0 m/s³):**
   Il rumore di jitter delle articolazioni 2D frame per frame si amplifica enormemente nella derivata seconda temporale, risultando **40 volte peggiore** rispetto a DELTAv2 (`1452.0` vs `36.6 m/s³`).
2. **Alto Tasso di Scambio di Identità (30.0%, 9/30 video):**
   Nelle viste a $90^\circ$ e $45^\circ$, OpenPose confonde frequentemente l'arto destro e sinistro durante l'incrocio delle gambe, trasferendo i punti sull'arto opposto.
3. **Collasso nelle Viste Assiali (0° e 180°):**
   Quando il soggetto cammina verso la telecamera o si allontana, l'accorciamento prospettico del femore in 2D genera stime di profondità errate, producendo un **tasso di artefatto assiale del 40.0%**.
