# 06 — Dummy Baseline Method (OpenPose 2D Lift)

This document provides the technical specification of the non-deep-learning geometric baseline ("Dummy Method") implemented in `src/inference/dummy_tracker.py` and configured in `configs/default_config.yaml`.

---

## 1. Motivation & Purpose

To quantitatively isolate the contribution of dense metric 3D point tracking (DELTAv2 + UniDepthV2) and geometric post-processing (`GeometricPruner`), the benchmark includes a purely 2D skeletal geometric baseline:
- Instead of tracking dense optical features in 3D using spatio-temporal attention, the **Dummy Method** relies strictly on sparse 2D skeletal keypoint detections from **OpenPose** (`RHip`, `RKnee`, `LHip`, `LKnee`).
- It tests a fundamental ablation question: *Can clinical 3D thigh kinematics be reliably extracted simply by rigidly transferring initial thigh points using 2D skeleton joints and a constant-depth pinhole lift?*

---

## 2. Mathematical Formulation & Point Transfer

### A. Anatomical Thigh Basis at the Reference Frame
At the optimal reference frame $t_{\text{ref}}$, three anatomical query points $P_1, P_2, P_3 \in \mathbb{R}^2$ are defined inside the thigh contour (from `ThighPointSelector` or manual validation).
Let the 2D knee joint coordinate be $\mathbf{p}_{\text{knee}} \in \mathbb{R}^2$ and the hip joint coordinate be $\mathbf{p}_{\text{hip}} \in \mathbb{R}^2$.
The longitudinal thigh vector $\mathbf{v}$ and its unit direction $\hat{\mathbf{u}}$ are:

$$\mathbf{v} = \mathbf{p}_{\text{hip}} - \mathbf{p}_{\text{knee}}, \qquad L_{\text{femur}} = \|\mathbf{v}\|_2, \qquad \hat{\mathbf{u}} = \frac{\mathbf{v}}{L_{\text{femur}}}$$

The transverse (orthogonal) unit vector $\hat{\mathbf{u}}_\perp$ is defined in 2D such that $\hat{\mathbf{u}} \cdot \hat{\mathbf{u}}_\perp = 0$:

$$\hat{\mathbf{u}}_\perp = \begin{bmatrix} -\hat{u}_y \\ \hat{u}_x \end{bmatrix}$$

### B. Scale-Invariant Coordinate Extraction
Each thigh point $P_i$ ($i \in \{1, 2, 3\}$) is parameterized into scale-invariant relative coordinates $(\alpha_i, \beta_i)$ relative to the knee joint:

$$\Delta\mathbf{p}_i = P_i - \mathbf{p}_{\text{knee}}, \qquad \alpha_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}}{L_{\text{femur}}}, \qquad \beta_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}_\perp}{L_{\text{femur}}}$$

### C. Frame-by-Frame Transfer Across the Video
For every target frame $t \in [0, T-1]$ where OpenPose detects knee $\mathbf{p}_{\text{knee}}(t)$ and hip $\mathbf{p}_{\text{hip}}(t)$ with sufficient confidence:

$$P_i(t) = \mathbf{p}_{\text{knee}}(t) + L_{\text{femur}}(t) \left( \alpha_i \hat{\mathbf{u}}(t) + \beta_i \hat{\mathbf{u}}_\perp(t) \right)$$

Missing or occluded joint detections are linearly interpolated across time.

---

## 3. Naive Pinhole 3D Lifting (`depth_mode = "constant"`)

To lift the transferred 2D points into 3D camera coordinates $(X, Y, Z)$ without a monocular depth network, `DummyTracker.lift_to_3d` uses a pinhole camera model with principal point $(c_x, c_y) = (W/2, H/2)$ and nominal focal length $f = 1.2 \cdot \max(W, H)$ (`focal_factor = 1.2`):

$$\mathbf{X}_{\text{cam}, i}(t) = \begin{bmatrix} \frac{P_{i, x}(t) - c_x}{f} \cdot Z(t) \\ \frac{P_{i, y}(t) - c_y}{f} \cdot Z(t) \\ Z(t) \end{bmatrix}$$

- **Production Benchmark Configuration (`depth_mode = "constant"`):** Uses a constant planar depth $Z(t) = Z_0 = 2.5\text{ m}$ (`default_depth_m = 2.5` in `configs/default_config.yaml`). Because all 3 points lie on the same fronto-parallel plane at constant $Z_0$ and are rigidly transferred in 2D, the resulting 3D pairwise distance variation is identically `0.0%` (**planar geometric tautology**, not genuine 3D rigidity).
- **Optional Mode (`depth_mode = "anthropometric"`):** Also implemented in `dummy_tracker.py` by scaling $Z(t) = Z_0 \cdot (L_{\text{femur}}(t_{\text{ref}}) / L_{\text{femur}}(t))$ (clamped to $[0.2, 5.0]$), but not used in the primary benchmark because foreshortening in axial views causes extreme depth divergence.

---

## 4. Benchmark Findings & Limitations (40 Gait360 Videos)

1. **Severe Acceleration Jitter (`RMS Jerk = 1452.0 ± 646.6 m/s³`):**
   Frame-to-frame OpenPose 2D keypoint jitter explodes under second-order differentiation (**~40× noisier** than DELTAv2 Refined Triplet, `36.6 ± 14.3 m/s³`).
2. **Severe Pitch ROM Underestimation (`-31.7% ± 46.0%`):**
   Compresses true thigh sagittal excursion by nearly one third compared to DELTAv2 Refined Triplet (`+4.1% ± 70.8%`).
3. **High Identity Swap Rate (`30.0%`, 9/30 crossover videos):**
   In $90^\circ$ and $45^\circ$ views, OpenPose frequently swaps left/right hip and knee labels during contralateral leg crossing (3× worse than DELTAv2 Refined Triplet's `10.0%`, 3/30).
4. **Axial View Collapse (`40.0%`, 4/10 frontal/posterior videos):**
   Without true 3D depth estimation, perspective foreshortening along the optical axis triggers antiphase artifacts in 4 out of 10 axial trials (vs `10.0%`, 1/10 for DELTAv2 Refined Triplet).
