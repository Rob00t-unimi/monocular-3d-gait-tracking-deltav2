# 04 — Mathematical Formulation & Kinematic Processing

This chapter provides the formal mathematical specification of the 3D post-processing pipeline, the spatial coordinate systems, the rigid-body kinematic alignment, and the signal-processing validation metrics.

---

## 1. Anisotropic Metric Tensor for Monocular Depth (`z_weight = 0.25`)

Monocular metric depth estimators (UniDepthV2) exhibit significantly higher uncertainty along the optical axis $Z$ than along the transverse image-plane axes $(X, Y)$ due to projective geometry and foreground-background edge ambiguity (Z-spikes).
In **Stage 1 (Spatial DBSCAN Clustering)** and **Stage 2 (Kinematic MAD Pruning)** of `GeometricPruner`, spatial distances and trajectory derivatives are computed under an anisotropic metric tensor $\mathbf{W}_z$:

$$\mathbf{W}_z = \text{diag}(1,\; 1,\; w_z), \qquad w_z = 0.25$$

The distance between two points $\mathbf{p}_a, \mathbf{p}_b \in \mathbb{R}^3$ is defined as:

$$d_{\mathbf{W}_z}(\mathbf{p}_a, \mathbf{p}_b) = \|\mathbf{W}_z(\mathbf{p}_a - \mathbf{p}_b)\|_2 = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2 + w_z^2(z_a - z_b)^2}$$

In Stage 2, velocity deviations from the median cluster profile are filtered using a Median Absolute Deviation (MAD) threshold with a physical noise floor $\sigma_{\min} = 0.015\text{ m/frame}$:

$$\tau_v = \max\!\left(2.5 \cdot 1.4826 \cdot \text{MAD}(\|\mathbf{W}_z(\mathbf{v}_i - \bar{\mathbf{v}})\|_2),\; \sigma_{\min}\right)$$

---

## 2. Geometric Conditioning of Rigid Body Triplets (Altitude & Aspect Ratio)

Given a candidate rigid triplet of 3D points $(\mathbf{p}_1(t), \mathbf{p}_2(t), \mathbf{p}_3(t)) \in \mathbb{R}^{3 \times 3}$, let the side vectors be $\mathbf{e}_{12} = \mathbf{p}_2 - \mathbf{p}_1$, $\mathbf{e}_{23} = \mathbf{p}_3 - \mathbf{p}_2$, and $\mathbf{e}_{31} = \mathbf{p}_1 - \mathbf{p}_3$, with maximum edge length $L_{\max}(t)$ and minimum edge length $L_{\min}(t)$.
The instantaneous triangle area $A(t)$ and minimum physical altitude (lever arm) $h_{\min}(t)$ are given by:

$$A(t) = \frac{1}{2}\|\mathbf{e}_{12}(t) \times (\mathbf{p}_3(t) - \mathbf{p}_1(t))\|_2, \qquad h_{\min}(t) = \frac{2A(t)}{L_{\max}(t)}$$

The unit surface normal vector $\hat{\mathbf{n}}(t)$ is defined as:

$$\hat{\mathbf{n}}(t) = \frac{\mathbf{e}_{12}(t) \times \mathbf{e}_{13}(t)}{\|\mathbf{e}_{12}(t) \times \mathbf{e}_{13}(t)\|_2}$$

When $h_{\min}$ is small (triangles approaching collinearity), minor depth variations $\delta z$ along the optical axis induce large spurious rotations $O(\delta z / h_{\min})$ of $\hat{\mathbf{n}}(t)$. To guarantee numerical conditioning of the thigh normal vector, Stage 3 enforces two geometric bounds evaluated both at the reference frame and as a temporal mean:

$$h_{\min} \ge 0.035\text{ m}\;(3.5\text{ cm}), \qquad \text{AR} = \frac{L_{\max}}{L_{\min}} \le 8.0$$

---

## 3. Anatomical Thigh Coordinate System and IMU Axis Alignment

To evaluate visual tracking against wearable inertial sensors without coordinate distortion:

1. **Thigh Anatomical Basis:** The visual femur orientation is tracked directly in 3D Euclidean space through the normal vector $\hat{\mathbf{n}}(t)$ and the longitudinal femur unit vector $\hat{\mathbf{u}}_{\text{femur}}(t) = \frac{\mathbf{p}_{\text{knee}}(t) - \mathbf{p}_{\text{hip}}(t)}{\|\mathbf{p}_{\text{knee}}(t) - \mathbf{p}_{\text{hip}}(t)\|_2}$. The sagittal flexion/extension angle (Pitch) is extracted from the rotation angle around the transverse anatomical axis.
2. **Preservation of Sensor Mounting Axes (`use_pca_basis = False`):** The WitMotion IMU sensor is attached to the anterior aspect of the right thigh. Its nominal axes correspond to:
   - $X$-axis: Longitudinal limb axis (pointing downward toward the knee).
   - $Y$-axis: Anteroposterior axis (orthogonal to the front surface of the thigh).
   - $Z$-axis: Mediolateral transverse axis.
   The nominal physical sensor axes are preserved directly without unsupervised PCA reorientation, ensuring direct physical correspondence with the anatomical limb frame.

---

## 4. Kabsch $\text{SO}(3)$ Rigid-Body Orientation & Velocity Integration

1. **Optimal $\text{SO}(3)$ Rigid Alignment:** To determine the static spatial transformation between the camera coordinate system and the inertial world frame at initial calibration, the system utilizes the Kabsch algorithm. Given zero-mean visual acceleration vectors $\mathbf{A}_{\text{vis}} \in \mathbb{R}^{T \times 3}$ and IMU acceleration vectors $\mathbf{A}_{\text{imu}} \in \mathbb{R}^{T \times 3}$, the cross-covariance matrix $\mathbf{H} = \mathbf{A}_{\text{vis}}^\top \mathbf{A}_{\text{imu}}$ is decomposed via Singular Value Decomposition:

   $$\mathbf{H} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

   The optimal proper rotation matrix $\mathbf{R}^* \in \text{SO}(3)$ satisfying $\det(\mathbf{R}^*) = +1$ is:

   $$\mathbf{R}^* = \mathbf{V}\,\text{diag}\!\left(1,\; 1,\; \det(\mathbf{V}\mathbf{U}^\top)\right)\mathbf{U}^\top$$

2. **Single-Pass Zero-Mean Velocity Integration:** Instantaneous 3D velocity $\mathbf{v}(t)$ is obtained from linear acceleration $\mathbf{a}(t)$ by single-pass cumulative trapezoidal integration:

   $$\mathbf{v}(t) = \int_0^t \mathbf{a}(\tau)\,d\tau - \frac{1}{T}\int_0^T \left(\int_0^t \mathbf{a}(\tau)\,d\tau\right)dt$$

   This ensures a drift-free, zero-mean velocity profile across the analyzed stride sequence.

---

## 5. Signed Waveform Cross-Correlation with Physiological Lag Cap

The temporal lag $\tau^*$ between the visually derived angular velocity $\omega_{\text{vis}}(t)$ and the IMU gyro signal $\omega_{\text{imu}}(t)$ is determined via normalized cross-correlation:

$$\rho(\tau) = \frac{\sum_{t} \left(\omega_{\text{vis}}(t) - \bar{\omega}_{\text{vis}}\right)\left(\omega_{\text{imu}}(t + \tau) - \bar{\omega}_{\text{imu}}\right)}{\sqrt{\sum_t \left(\omega_{\text{vis}}(t) - \bar{\omega}_{\text{vis}}\right)^2 \sum_t \left(\omega_{\text{imu}}(t + \tau) - \bar{\omega}_{\text{imu}}\right)^2}}$$

1. **Signed Preservation:** Cross-correlation is computed on signed waveforms (preserving positive and negative values). This ensures that anti-phase tracking errors (such as $180^\circ$ phase flips or contralateral identity swaps) produce negative correlation coefficients rather than false positive scores.
2. **Physiological Lag Bound:** The search window for the peak alignment lag is strictly bounded within a physiological window of $\pm 0.5\text{ s}$ around nominal synchronization:

$$\tau^* = \arg\max_{|\tau| \le 0.5\,\text{s}} \rho(\tau)$$
