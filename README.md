# Flow Matching-based Lightweight Pipeline for 3D Shape Generation in Edge Computing Environments

An optimized, CPU-friendly 3D asset generation pipeline designed for ARM-based edge devices (e.g., Raspberry Pi 5). This repository implements the official code for the paper: *"A Study on the Implementation of a Flow Matching-based Lightweight Pipeline for 3D Shape Generation in Edge Computing Environments."*

---

## 🚀 Key Contributions
* **ARM-CPU Compatibility (einsum Patch):** Solved PyTorch `nn.Linear` execution failures on ARM architectures (caused by MKL-DNN matmul primitive constraints with 3D tensors) by replacing linear projections with robust `torch.einsum` operations.
* **Fast Sampling with OT-CFM:** Replaced the legacy curved, multi-step trajectories of EDM with **Optimal Transport Conditional Flow Matching (OT-CFM)**. This straightens the generation path, dramatically reducing the required sampling steps from **18 to 8** without losing structural diversity.
* **Edge-Deployable Efficiency:** Achieved a **2.4x speedup** on hardware with limited resources, showing stable operations within ~30% of total memory on a Raspberry Pi 5.

---

## 🛠️ System Architecture
The generation pipeline consists of two primary stages:
1. **Flow Matching Sampling:** Generates a category-conditioned latent vector ($z \in \mathbb{R}^{32 \times 32}$) from pure noise ($z_0 \sim \mathcal{N}(0, I)$) using a Transformer backbone with cross-attention.
2. **ARM-Based COD-VAE Decoding:** Decodes the sampled latent vector into a tri-plane representation, transforms it into a $128^3$ occupancy field, and extracts the final 3D mesh via the Marching Cubes algorithm.

---

## 📊 Experimental Results

### 1. Generation Speed & Efficiency (Measured on Raspberry Pi 5)
By introducing OT-CFM, the total generation loop is shortened significantly while keeping the VAE decoder stage consistent.

| Stage | Legacy (EDM) | **Ours (OT-CFM)** | **Ratio (Speedup)** |
| :--- | :---: | :---: | :---: |
| **Noise Sampling** | $19.77\text{s } (\pm0.20)$ | **$4.50\text{s } (\pm0.05)$** | **$4.4\times$** |
| **VAE Decoder** | $6.91\text{s } (\pm0.13)$ | $6.70\text{s } (\pm0.23)$ | $1.0\times$ |
| **Total Time** | $26.79\text{s } (\pm0.21)$ | **$11.34\text{s } (\pm0.24)$** | **$2.4\times$** |

### 2. Qualitative Quality & Resources
* **Qualitative Performance:** While quantitative metrics (MMD-CD, 1-NNA-CD) showed minor shifts due to reduced sampling steps, the visual structure across categories (`chair`, `airplane`, `table`, `car`, `rifle`) remains identical to the original high-fidelity outputs.
* **Resource Profile:** * VAE Reconstruction: Memory **1.8 GB** | CPU Usage **25.4%**
  * Full OT-CFM Pipeline: Memory **2.4 GB** | CPU Usage **79.7%**

---

## ⚙️ Environment Specs & Dependencies
* **Training / Bulk Generation:** NVIDIA H200 GPU
* **Edge Benchmarking Target:** Raspberry Pi 5 (ARM Cortex-A76, 8GB)
* **Dataset:** ShapeNet-Core (52 Categories Trained, 5 Core Categories Evaluated)

### Prerequisites
```bash
pip install torch torchvision mne-python trimesh scipy