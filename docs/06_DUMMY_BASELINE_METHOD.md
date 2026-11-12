# 06 — Dummy Baseline Method (OpenPose 2D Lift)

This document provides a detailed technical specification of the geometric baseline ("Dummy Method") implemented in `src/inference/dummy_tracker.py` and evaluated across the Gait360 benchmark.

---

## 1. Motivation & Purpose

To quantitatively isolate the contribution of deep metric 3D point tracking (DELTAv2) and geometric post-processing (GeometricPruner), the benchmark includes a non-deep-learning 2D geometric baseline:
- Instead of tracking arbitrary optical features across time using cross-attention, the **Dummy Method** relies strictly on sparse 2D skeletal keypoint detections from **OpenPose** (`RHip`, `RKnee`, `LHip`, `LKnee`).
- It tests a critical scientific baseline question: *Can clinical thigh kinematics be reliably extracted simply by projecting and transferring initial thigh points according to 2D skeleton joints?*

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

$$\Delta\mathbf{p}_i = P_i - \mathbf{p}_{\text{knee}}$$
$$\alpha_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}}{L_{\text{femur}}}, \qquad \beta_i = \frac{\Delta\mathbf{p}_i \cdot \hat{\mathbf{u}}_\perp}{L_{\text{femur}}}$$

Here, $\alpha_i$ represents the normalized longitudinal position along the femur ($0 = \text{knee}$, $1 = \text{hip}$), while $\beta_i$ represents the transverse deviation.

### C. Frame-by-Frame Transfer Across the Video
For every target frame $t \in [0, T-1]$ where OpenPose detects knee $\mathbf{p}_{\text{knee}}(t)$ and hip $\mathbf{p}_{\text{hip}}(t)$ with sufficient confidence:
1. Recompute the instantaneous femur length $L_{\text{femur}}(t)$, longitudinal unit vector $\hat{\mathbf{u}}(t)$, and orthogonal unit vector $\hat{\mathbf{u}}_\perp(t)$.
2. Transfer the relative coordinates back to image space:

$$P_i(t) = \mathbf{p}_{\text{knee}}(t) + L_{\text{femur}}(t) \left( \alpha_i \hat{\mathbf{u}}(t) + \beta_i \hat{\mathbf{u}}_\perp(t) \right)$$

If joint detections are missing or occluded at frame $t$, coordinates are linearly interpolated.

---

## 3. Naive Pinhole 3D Lifting

To project the transferred 2D points into camera space $(X, Y, Z)$ without a dense metric depth model, the Dummy Method applies a naive geometric lifting:
1. **Depth from Anthropometric Femur Prior:** Assuming an average human femur length $L_{\text{phys}} \approx 0.42\text{ m}$ and camera focal length $f$ (from camera intrinsics), the instantaneous estimated depth $\tilde{Z}(t)$ is approximated by:

   $$\tilde{Z}(t) = \frac{f \cdot L_{\text{phys}}}{L_{\text{femur}}(t)}$$

2. **Inverse Pinhole Projection:**

   $$\mathbf{X}_{\text{cam}, i}(t) = \begin{bmatrix} \frac{P_{i, x}(t) - c_x}{f_x} \cdot \tilde{Z}(t) \\ \frac{P_{i, y}(t) - c_y}{f_y} \cdot \tilde{Z}(t) \\ \tilde{Z}(t) \end{bmatrix}$$

---

## 4. Benchmark Findings & Limitations

When evaluated across the 40 Gait360 videos:
1. **Severe Acceleration Jitter (RMS Jerk = 1452.0 m/s³):**
   Frame-to-frame OpenPose joint jitter induces massive second-derivative acceleration noise (**40× higher** than DELTAv2 Refined Triplet, `1452.0` vs `36.6 m/s³`).
2. **High Identity Swap Rate (30.0%, 9/30 crossover videos):**
   In $90^\circ$ and $45^\circ$ views, OpenPose frequently confuses the left and right knee/hip during contralateral limb crossing, transferring the points onto the wrong leg.
3. **Axial View Collapse (0° and 180°):**
   When the subject walks toward or away from the camera, 2D femur shortening causes erratic depth estimates, producing a **40.0% axial depth artifact rate**.
