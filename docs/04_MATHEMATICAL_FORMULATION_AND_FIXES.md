# 04 — Mathematical Formulation & Kinematic Processing

This chapter provides the formal mathematical specification of the 3D post-processing pipeline (`src/postprocessing/geometric_pruner.py`), camera-to-world kinematic alignment (`src/analysis/kinematic_alignment.py`), and swap disentanglement metrics (`src/metrics/`), aligned with the codebase and Technical Report.

---

## 1. Zero-Phase Adaptive Filtering & 2D/3D Geometric Pruning (`GeometricPruner` v4.1)

Given raw metric trajectories $\mathbf{X}(t) \in \mathbb{R}^{N_{\text{thigh}} \times 3}$ and 2D tracks $\mathbf{x}(t) \in \mathbb{R}^{N_{\text{thigh}} \times 2}$ for the $N_{\text{thigh}} = 203$ thigh points (`indices 0..202`):

1. **Zero-Phase Butterworth Low-Pass Pre-Filtering:**
   All 2D and 3D channels are filtered forward-backward (`scipy.signal.filtfilt`, 4th-order Butterworth) at the dataset-wide physiological cutoff $f_c = 2.9296875\text{ Hz}$ estimated via Welch PSD profiling ($f_{\text{low}} = 0.50\text{ Hz}$, $f_{\text{high}} \approx 3 \cdot f_{\text{median}}$).

2. **2D Convex Hull Area Persistence Pruning:**
   At each frame $t \in [0, T-1]$, let $A_{\text{hull}}(t)$ be the 2D Convex Hull area of the active thigh cluster and $A_{\text{hull}}(t_{\text{ref}})$ its area at the reference frame. Whenever $A_{\text{hull}}(t) > \gamma_{\max}\,A_{\text{hull}}(t_{\text{ref}})$ (`max_area_ratio = 5.0`), the farthest peripheral outlier is iteratively removed across **all** frames simultaneously, guaranteeing **100% temporal persistence**.

3. **Kinematic Velocity PCA + Adaptive DBSCAN (Coherent Cluster Isolation):**
   Point velocity profiles are standardized and projected onto a 5-component PCA embedding (`pca_components = 5`), then clustered via adaptive DBSCAN starting at `initial_eps = 0.5` and scaling by `1.3` (up to `15` steps) until a dominant coherent rigid-body cluster is isolated.

---

## 2. Combinatorial Optimal Rigid Triplet Selection

From the surviving thigh inlier set $\mathcal{I}$ (subsampled to at most `80` candidates if larger), candidate triplets $\tau = (i, j, k) \in \binom{\mathcal{I}}{3}$ are pre-filtered by $Z$-spike-regularized 3D reference area:

$$\tilde{A}_{3\text{D}}(\tau, t_{\text{ref}}) = \frac{A_{3\text{D}}(\tau, t_{\text{ref}})}{1 + w_z \left(\frac{\Delta z_{\max}}{\Delta xy_{\max} + \epsilon}\right)^2}$$

where $w_z$ (`z_spike_penalty_weight = 2.5` in config) penalizes triplets inflated by monocular optical-axis depth spikes. Among the top $K = 3000$ pre-filtered candidates, the algorithm searches adaptive non-collinearity floors $\rho_{\min} \in \{0.05, 0.02, 0.01, 0.005, 0.001\}$ enforcing $\min_t A_{3\text{D}}(\tau, t) \ge \rho_{\min}\,A_{3\text{D}}^{\text{orig}}(t_{\text{ref}})$, and maximizes the composite area-stability score:

$$\mathcal{S}(\tau) = \ln\!\bigl(1 + A_{2\text{D}}(\tau, t_{\text{ref}})\bigr)\,\ln\!\bigl(1 + \min_t A_{2\text{D}}(\tau, t)\bigr)\cdot \text{Stab}_{2\text{D}}(\tau)\,\text{Stab}_{3\text{D}}(\tau)$$

where $\text{Stab}_{d\text{D}}(\tau) = \mu(A_{d\text{D}}(\tau, \cdot)) / [\sigma(A_{d\text{D}}(\tau, \cdot)) + 10^{-6}]$ for $d \in \{2, 3\}$.

---

## 3. Camera-Space Kinematics & Tangent-Plane Twist Regularization

For a tracked triplet $(\mathbf{X}_t^1, \mathbf{X}_t^2, \mathbf{X}_t^3)$ (or $M$-point cloud):

1. **Centroid Linear Acceleration (Savitzky–Golay):**
   The instantaneous 3D centroid $\mathbf{c}_t = \frac{1}{M}\sum_{m=1}^{M}\mathbf{X}_t^m$ preserves 100% of full 3D metric coordinates. Its linear acceleration $\mathbf{a}_{\text{cam}}(t)$ is computed via Savitzky–Golay second-order differentiation (`window_length = 11`, `polyorder = 2`, `deriv = 2`).

2. **Tangent-Plane Twist Regularization ($\alpha_z = 0.05$) & Gram–Schmidt Orientation:**
   Because a 3-point patch on the thigh has median triangle altitude (lever arm) $h = 2.52\text{ cm}$ (IQR $[2.27, 3.27]\text{ cm}$), comparable to relative monocular depth jitter $\delta z = \sigma(Z_t^m - \bar{Z}_t)$ (median $1.37\text{ cm}$), unregularized depth noise induces parasitic normal deflections $\Delta\theta_{\mathbf{w}} \approx \arctan(1.37 / 2.52) \approx 28.5^\circ$ (peaks $45^\circ\text{--}90^\circ$). Before Gram–Schmidt orthogonalization, we attenuate only internal relative depth offsets around the instantaneous mean depth $\bar{Z}_t = \frac{1}{3}\sum_{m=1}^{3} Z_t^m$:

   $$\tilde{Z}_t^m = \bar{Z}_t + \alpha_z\,(Z_t^m - \bar{Z}_t), \qquad \alpha_z = 0.05$$

   and construct $\mathbf{R}_{\text{cam}}(t) = [\mathbf{u}_t, \mathbf{v}_t, \mathbf{w}_t] \in \text{SO}(3)$ on $\tilde{\mathbf{X}}_t^m = (X_t^m, Y_t^m, \tilde{Z}_t^m)^\top$:

   $$\mathbf{u}_t = \frac{\tilde{\mathbf{X}}_t^2 - \tilde{\mathbf{X}}_t^1}{\|\tilde{\mathbf{X}}_t^2 - \tilde{\mathbf{X}}_t^1\|_2}, \qquad \mathbf{w}_t = \frac{\mathbf{u}_t \times (\tilde{\mathbf{X}}_t^3 - \tilde{\mathbf{X}}_t^1)}{\|\mathbf{u}_t \times (\tilde{\mathbf{X}}_t^3 - \tilde{\mathbf{X}}_t^1)\|_2}, \qquad \mathbf{v}_t = \mathbf{w}_t \times \mathbf{u}_t$$

   enforcing temporal sign continuity $\mathbf{w}_t \cdot \mathbf{w}_{t-1} \ge 0$. This reduces median angular jitter to $1.56^\circ$ while preserving $\bar{Z}_t$ untouched. For multi-point clouds (*Refined Cloud SVD*), $\mathbf{w}_t$ ($w_{t,z} \ge 0$) and $\mathbf{y}_t$ are the smallest and largest right singular vectors of the centered cloud.

---

## 4. IMU World Projection, Kabsch $\text{SO}(3)$ Alignment & $\mathfrak{so}(3)$ Calibration

1. **IMU World Frame & Static Gravity Removal:**
   Using sensor orientation quaternions $\mathbf{R}_{\text{imu}}(t)$, body-frame specific force is rotated into the static World frame and the mean gravity vector is removed:

   $$\mathbf{a}_{\text{world, IMU}}(t) = \mathbf{R}_{\text{imu}}(t)\,\mathbf{a}_{\text{body}}(t) - \bar{\mathbf{g}}$$

2. **Bounded Overlap-Normalized Temporal Synchronization:**
   Clock offset $\Delta\tau$ between RGB video and wireless Bluetooth IMU is estimated by maximizing overlap-normalized cross-correlation within a bounded search window $|\tau| \le 2.5\text{ s}$ (`max_lag_seconds = 2.5`).

3. **Static Camera-to-World Rotation ($\mathbf{R}_{\text{cam}\to\text{world}}$):**
   Estimated via Kabsch SVD (`scipy.spatial.transform.Rotation.align_vectors`) aligning bandpass-filtered 3D visual acceleration $\mathbf{a}_{\text{cam}}(t)$ to $\mathbf{a}_{\text{world, IMU}}(t)$ in $\text{SO}(3)$.

4. **Anatomical PCA Basis & Relative Orientation in $\mathfrak{so}(3)$:**
   Because the thigh triangle plane and the wearable IMU housing belong to the same rigid segment but differ by an unknown static offset, relative rotations $\Delta\mathbf{R}(t) = \mathbf{R}(t_{\text{ref}})^\top \mathbf{R}(t)$ are mapped to the Lie algebra $\mathfrak{so}(3)$ via the logarithmic map; anatomical principal axes are calibrated via PCA (`use_anatomical_pca_basis = True`) so that sagittal flexion-extension Pitch is consistently aligned across arbitrary 360° viewpoints.

---

## 5. Biomechanical Swap Disentanglement

Over sliding windows of $1.2\text{ s}$ (step $0.2\text{ s}$), persistent rolling antiphase ($r(t) < -0.35$ for $\ge 35\%$ of windows) is classified by camera geometry:
- **Crossover Views ($90^\circ$ Sagittal & $45^\circ$ Oblique):** Classified as a **True Identity Swap** (tracker jump onto the contralateral limb during 2D occlusion).
- **Axial Views ($0^\circ$ Frontal & $180^\circ$ Posterior):** Left and right limbs remain horizontally separated by pelvic width and never intersect in 2D; persistent antiphase is classified as an **Axial Monocular Depth Artifact**.
