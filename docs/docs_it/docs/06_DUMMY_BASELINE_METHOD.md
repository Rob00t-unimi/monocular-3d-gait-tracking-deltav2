# 06 — Metodo Baseline Dummy (Lifting 2D OpenPose)

Questo documento fornisce la specifica tecnica della baseline geometrica non deep-learning ("Metodo Dummy") implementata in `src/inference/dummy_tracker.py` e configurata in `configs/default_config.yaml`.

---

## 1. Motivazione e Scopo

Per isolare quantitativamente il contributo del tracking 3D metrico denso (DELTAv2 + UniDepthV2) e del post-processing geometrico (`GeometricPruner`), il benchmark include una baseline puramente geometrica 2D:
- Invece di tracciare feature ottiche dense in 3D con attenzione spazio-temporale, il **Metodo Dummy** si affida esclusivamente ai keypoint scheletrici 2D sparsi di **OpenPose** (`RHip`, `RKnee`, `LHip`, `LKnee`).
- Verifica una domanda scientifica fondamentale: *È possibile estrarre una cinematica clinica 3D della coscia affidabile semplicemente trasferendo rigidamente i punti iniziali secondo i giunti 2D e proiettandoli a profondità costante?*

---

## 2. Formulazione Matematica e Trasferimento dei Punti

### A. Base Anatomica Femorale al Frame di Riferimento
Al frame di riferimento ottimale $t_{\text{ref}}$, i tre punti di ancoraggio $P_1, P_2, P_3 \in \mathbb{R}^2$ sono definiti all'interno del contorno della coscia.
Siano $\mathbf{p}_{\text{knee}} \in \mathbb{R}^2$ il ginocchio e $\mathbf{p}_{\text{hip}} \in \mathbb{R}^2$ l'anca. Il vettore longitudinale femorale $\mathbf{v}$ e il versore $\hat{\mathbf{u}}$ sono:

$$\mathbf{v} = \mathbf{p}_{\text{hip}} - \mathbf{p}_{\text{knee}}, \qquad L_{\text{femur}} = \|\mathbf{v}\|_2, \qquad \hat{\mathbf{u}} = \frac{\mathbf{v}}{L_{\text{femur}}}$$

Il versore trasversale (ortogonale) $\hat{\mathbf{u}}_\perp$ in 2D è:

$$\hat{\mathbf{u}}_\perp = \begin{bmatrix} -\hat{u}_y \\ \hat{u}_x \end{bmatrix}$$

### B. Coordinate Relative Invarianti di Scala
Ogni punto $P_i$ ($i \in \{1, 2, 3\}$) viene parametrizzato in coordinate locali $(\alpha_i, \beta_i)$ rispetto al ginocchio:

$$\Delta\mathbf{p}_i = P_i - \mathbf{p}_{\text{knee}}, \qquad \alpha_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}}{L_{\text{femur}}}, \qquad \beta_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}_\perp}{L_{\text{femur}}}$$

### C. Trasferimento Frame-per-Frame sul Video
Per ogni frame $t \in [0, T-1]$ con rilevamenti validi di ginocchio e anca:

$$P_i(t) = \mathbf{p}_{\text{knee}}(t) + L_{\text{femur}}(t) \left( \alpha_i \hat{\mathbf{u}}(t) + \beta_i \hat{\mathbf{u}}_\perp(t) \right)$$

Eventuali frame con giunti mancanti o occlusi vengono interpolati linearmente nel tempo.

---

## 3. Lifting 3D Pinhole Ingenuo (`depth_mode = "constant"`)

Per sollevare i punti 2D in coordinate camera 3D $(X, Y, Z)$ senza rete di profondità monoculare, `DummyTracker.lift_to_3d` applica un modello pinhole con punto principale $(c_x, c_y) = (W/2, H/2)$ e focale nominale $f = 1.2 \cdot \max(W, H)$ (`focal_factor = 1.2`):

$$\mathbf{X}_{\text{cam}, i}(t) = \begin{bmatrix} \frac{P_{i, x}(t) - c_x}{f} \cdot Z(t) \\ \frac{P_{i, y}(t) - c_y}{f} \cdot Z(t) \\ Z(t) \end{bmatrix}$$

- **Configurazione del Benchmark (`depth_mode = "constant"`):** Utilizza una profondità planare costante $Z(t) = Z_0 = 2.5\text{ m}$ (`default_depth_m = 2.5` in `configs/default_config.yaml`). Poiché i 3 punti giacciono sullo stesso piano fronto-parallelo a $Z_0$ costante e vengono trasferiti rigidamente in 2D, la variazione relativa delle distanze 3D è identicamente `0.0%` (**tautologia geometrica planare**, non rigidità 3D reale).
- **Modalità Opzionale (`depth_mode = "anthropometric"`):** Implementata in `dummy_tracker.py` scalando $Z(t) = Z_0 \cdot (L_{\text{femur}}(t_{\text{ref}}) / L_{\text{femur}}(t))$ (limitata in $[0.2, 5.0]$), ma non usata nel benchmark primario poiché lo scorcio nelle viste assiali provoca divergenze estreme di profondità.

---

## 4. Risultati del Benchmark e Limiti (40 Video Gait360)

1. **Grave Jitter di Accelerazione (`RMS Jerk = 1452.0 ± 646.6 m/s³`):**
   Il jitter dei keypoint 2D OpenPose esplode alla derivazione seconda (**~40× più rumoroso** rispetto a DELTAv2 Refined Triplet, `36.6 ± 14.3 m/s³`).
2. **Forte Sottostima del ROM sul Pitch (`-31.7% ± 46.0%`):**
   Comprime l'escursione articolare sagittale di quasi un terzo rispetto a DELTAv2 Refined Triplet (`+4.1% ± 70.8%`).
3. **Elevato Tasso di Scambi d'Identità (`30.0%`, 9/30 video con incrocio):**
   Nelle viste a $90^\circ$ e $45^\circ$, OpenPose confonde frequentemente anca/ginocchio destro e sinistro durante il passo incrociato (3× peggiore rispetto al `10.0%`, 3/30 di DELTAv2 Refined Triplet).
4. **Collasso sulle Viste Assiali (`40.0%`, 4/10 video frontali/posteriori):**
   Senza stima reale della profondità 3D, lo scorcio prospettico lungo l'asse ottico genera artefatti in antifase su 4 prove assiali su 10 (contro `10.0%`, 1/10 per DELTAv2 Refined Triplet).
