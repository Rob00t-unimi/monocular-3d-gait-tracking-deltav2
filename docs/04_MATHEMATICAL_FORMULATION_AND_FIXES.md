# 04 — Mathematical Formulation & Critical Algorithmic Fixes

This chapter provides the formal mathematical specification of the 3D post-processing, kinematic alignment, and validation metrics, including the five critical mathematical fixes validated across the 40-video benchmark.

---

## 1. Anisotropic Monocular Depth Scaling (`z_weight = 0.25`)

Monocular metric depth estimators (UniDepthV2) exhibit higher variance along the optical axis $Z$ than along image-plane axes $(X, Y)$ due to foreground-background boundary bleeding (ray-shooting Z-spikes).
In **Stage 1 (Spatial DBSCAN)** and **Stage 2 (Kinematic MAD Pruning)** of `GeometricPruner`, distances and kinematic derivatives are evaluated under the anisotropic metric tensor:

$$\mathbf{W}_z = \text{diag}(1,\; 1,\; w_z), \qquad w_z = 0.25$$

Furthermore, Stage 2 velocity deviation thresholds enforce a physical noise floor $\sigma_{\min} = 0.015\text{ m/frame}$ so that low-jitter trajectories are never over-pruned when $\text{MAD}_v \to 0$:

$$\tau_v = \max\!\left(2.5 \cdot 1.4826 \cdot \text{MAD}(\|\mathbf{W}_z(\mathbf{v}_i - \bar{\mathbf{v}})\|_2),\; 0.015\right)$$

---

## 2. Triangle Altitude & Aspect-Ratio Guard Against Normal-Vector Twist

Given a candidate rigid triplet $(\mathbf{p}_1(t), \mathbf{p}_2(t), \mathbf{p}_3(t)) \in \mathbb{R}^{3 \times 3}$, let the edge vectors be $\mathbf{e}_{12} = \mathbf{p}_2 - \mathbf{p}_1$, $\mathbf{e}_{23} = \mathbf{p}_3 - \mathbf{p}_2$, $\mathbf{e}_{31} = \mathbf{p}_1 - \mathbf{p}_3$, with maximum edge length $L_{\max}$ and minimum edge length $L_{\min}$.
The 3D triangle area and minimum physical altitude (lever arm) are:

$$A(t) = \frac{1}{2}\|\mathbf{e}_{12}(t) \times (\mathbf{p}_3(t) - \mathbf{p}_1(t))\|_2, \qquad h_{\min}(t) = \frac{2A(t)}{L_{\max}(t)}$$

If $h_{\min}$ is small (nearly collinear points), a minor depth perturbation $\delta z$ causes an $O(\delta z / h_{\min})$ angular twist of the surface normal $\hat{\mathbf{n}}(t)$, corrupting 3D orientation and Range of Motion (ROM).
Stage 3 enforces two hard geometric conditioning bounds on the reference frame and temporal mean:

$$h_{\min} \ge 0.035\text{ m}\;(3.5\text{ cm}), \qquad \text{AR} = \frac{L_{\max}}{L_{\min}} \le 8.0$$

**Empirical Impact:** Reduced Pitch ROM Error on the 40-video benchmark from `+45.2% ± 112.9%` (Pruned Cloud) down to **`+4.1% ± 70.8%`** (Refined Triplet) and improved Pitch Pearson $r$ to **`0.295 ± 0.329`**.

---

## 3. Anatomical Thigh Coordinate Frame (`use_pca_basis = False`) & Removal of SVD Camera Leak

1. **Removal of Global SVD Plane Projection Leak:** Earlier prototypes projected 3D trajectories onto the 2 principal components of the global trajectory matrix via `np.linalg.svd` inside `SignalExtractor`, mixing sagittal flexion with frontal abduction on $45^\circ$ oblique views. This projection was completely removed.
2. **Preservation of the Anatomical IMU Frame:** Applying an unsupervised PCA rotation (`use_pca_basis=True`) to the WitMotion IMU accelerometer channels scrambled the physical longitudinal thigh axis ($X$) whenever mediolateral acceleration variance was comparable to longitudinal variance. Setting `use_pca_basis=False` by default preserves the physical sensor axes prior to Kabsch $\text{SO}(3)$ alignment.

---

## 4. Kabsch $\text{SO}(3)$ Rigid Alignment & Honest Single-Pass Velocity Integration

Given zero-mean vision specific-force vectors $\mathbf{A}_{\text{vis}} \in \mathbb{R}^{T \times 3}$ and IMU specific-force vectors $\mathbf{A}_{\text{imu}} \in \mathbb{R}^{T \times 3}$, the optimal proper rotation $\mathbf{R}^* \in \text{SO}(3)$ is obtained via Singular Value Decomposition of the cross-covariance matrix $\mathbf{H} = \mathbf{A}_{\text{vis}}^\top \mathbf{A}_{\text{imu}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$:

$$\mathbf{R}^* = \mathbf{V}\,\text{diag}\!\left(1,\; 1,\; \det(\mathbf{V}\mathbf{U}^\top)\right)\mathbf{U}^\top$$

Integrated 3D velocities are computed via a single zero-mean cumulative trapezoidal integration pass without post-hoc re-centering leakage.

---

## 5. Signed Temporal Cross-Correlation with Physiological Lag Cap ($\pm 0.5\text{ s}$)

Replacing the unsigned Euclidean norm (which rectifies bipolar oscillations and doubles the fundamental frequency from $f_0$ to $2f_0$, blinding phase inversion checks) with signed channel cross-correlation bounded to a physiological lag window $|\Delta t| \le 0.5\text{ s}$ ensures honest phase and lag evaluation across all 40 videos.
