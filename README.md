# 🎨 Real-ESRGAN Anime Wallpaper AI Upscaler

A high-performance, **100% local, offline AI super-resolution pipeline** designed to upscale, denoise, and restore anime wallpapers, digital artwork, and smartphone backgrounds to crisp 4K/8K resolutions.

Built on modern **PyTorch**, accelerated via **NVIDIA CUDA (Tensor Cores / FP16)**, and powered by the lightweight **`RealESRGAN_x4plus_anime_6B`** architecture via **Spandrel**, this repository eliminates legacy dependencies (`basicsr`, unmaintained binaries) in favor of a clean, robust, and future-proof implementation.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [How Real-ESRGAN Works (Architecture Deep Dive)](#-how-real-esrgan-works-architecture-deep-dive)
  - [The Convolutional Backbone: RRDBNet](#1-the-convolutional-backbone-rrdbnet)
  - [Why the &#34;Anime 6B&#34; Variant?](#2-why-the-anime-6b-variant)
  - [High-Order Degradation Modeling](#3-high-order-degradation-modeling)
  - [Seamless Tiled Inference](#4-seamless-tiled-inference)
- [Repository Structure](#-repository-structure)
- [Installation &amp; Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [Mode 1: Automated Batch Processing (`anime_upscaler.py`)](#mode-1-automated-batch-processing-anime_upscalerpy)
  - [Mode 2: Interactive Web Studio (`app_wallpaper_upscaler.py`)](#mode-2-interactive-web-studio-app_wallpaper_upscalerpy)
- [Configuration &amp; Hardware Tuning](#-configuration--hardware-tuning)
- [Troubleshooting &amp; FAQ](#-troubleshooting--faq)

---

## 🌟 Key Features

- **100% Local & Private:** All inference runs on your local machine; zero images or telemetry ever leave your system.
- **Hardware Acceleration:** Native PyTorch CUDA acceleration with FP16 (Half-Precision) tensor operations, doubling execution speed and halving VRAM requirements.
- **Automated Aspect-Ratio Sorting:** Inspects source image dimensions and routes outputs into dedicated subfolders (`desktop_horizontal` vs. `mobile_vertical`).
- **Dynamic VRAM Tiling:** Automatically breaks ultra-high-resolution images into overlapping tiles on the GPU and stitches them back together, preventing CUDA Out-Of-Memory (OOM) errors even with 8K outputs.
- **Automated Weights Management:** Downloads validated weights directly from Hugging Face on the first run; runs completely offline afterward.
- **Dual Interfaces:** Headless CLI batch pipeline for processing entire wallpaper libraries, plus an interactive Gradio Web UI featuring real-time before/after comparison sliders and device aspect-ratio fitting.

---

## 🧠 How Real-ESRGAN Works (Architecture Deep Dive)

Traditional image upscaling algorithms (e.g., Bilinear, Bicubic, or Lanczos) calculate intermediate pixel values through mathematical interpolation. While fast, interpolation cannot synthesize missing high-frequency details; it inherently blurs edges and amplifies existing JPEG compression artifacts.

Real-ESRGAN (_Real-World Enhanced Super-Resolution Generative Adversarial Networks_) uses a **Deep Convolutional Neural Network (CNN)** trained to predict missing high-frequency textures from low-resolution inputs.

```
Input (LR Image)
       │
       ▼
[ Shallow Feature Extraction (3x3 Conv) ]
       │
       ▼
[ Residual-in-Residual Dense Blocks (RRDB) x 6 ]  ◄── Deep Feature Extraction
       │   ├── Dense Connections (Feature Reuse)
       │   └── Residual Scaling (Gradient Stability)
       ▼
[ Trunk Global Residual Connection ]
       │
       ▼
[ Upsampling Layer (Nearest Neighbor + Conv) ] ──► 4x Spatial Upscaling
       │
       ▼
[ High-Resolution Output Reconstruction ]
       │
       ▼
Output (4x HR Denoised Image)
```

### 1. The Convolutional Backbone: RRDBNet

At the core of the model is **RRDBNet** (_Residual-in-Residual Dense Block Network_):

- **No Batch Normalization (BN):** Standard CNNs use Batch Normalization to stabilize training, but BN frequently causes unpleasant halo artifacts and limits dynamic range in image super-resolution tasks. Real-ESRGAN removes all BN layers.
- **Dense Connections:** Inside each block, every convolutional layer receives features from all preceding layers. This multi-level feature reuse preserves fine structural lines across deep layers.
- **Residual-in-Residual (RIR) Structure:** Multiple Dense Blocks are grouped inside a Residual Block, which itself sits inside a global residual path. This hierarchical skip-connection structure allows gradients to flow smoothly during backward passes, enabling deep representation without vanishing gradients.

### 2. Why the "Anime 6B" Variant?

Standard Real-ESRGAN models for photographic images use a heavy **23-block** RRDBNet backbone. However, this repository employs the specialized **`RealESRGAN_x4plus_anime_6B`** model:

- **Compact Depth (6 Blocks):** Anime, digital illustrations, and manga line art feature distinct flat color fields, solid gradients, and well-defined line strokes, unlike complex, chaotic real-world textures (such as grass, skin pores, or fabric).
- **Artifact Suppression:** 23-block photo models often over-hallucinate high-frequency texture onto flat anime surfaces, creating unwanted painterly noise. The 6-block architecture cleanly eliminates JPEG blocking and ringing artifacts while preserving crisp line art.
- **Low Latency & Memory:** A 6-block network has roughly one-fourth the parameter footprint of the standard model, delivering faster inference times and lower VRAM consumption.

### 3. High-Order Degradation Modeling

Real-ESRGAN was trained using a synthetic **"high-order degradation"** scheme. Instead of assuming images only suffer from clean downsampling, the training process simulates real-world internet degradations: repeated cycles of blur filters, generalized Gaussian noise, downsampling, and aggressive JPEG compression. Consequently, the model excels at taking heavily compressed 1080p web wallpapers and restoring them to clean 4K.

### 4. Seamless Tiled Inference

Upscaling a 1080p or 1440p image directly by 4x produces an output canvas of up to 5760×3240 or 7680×4320 pixels. Running full-frame forward passes at this resolution on intermediate feature maps would exhaust the VRAM of standard consumer GPUs.

To guarantee zero out-of-memory crashes:

1. The input image is divided into spatial tiles (e.g., $1024 \times 1024$ pixels).
2. Each tile is padded with an overlap border (e.g., $32$ pixels) using context from adjacent tiles.
3. The CNN forward pass runs on individual padded tiles.
4. The padding margin is cropped out on the upscaled tile, and all tiles are seamlessly recombined. This eliminates edge-discontinuity seams between tile boundaries.

---

## 📁 Repository Structure

```text
RealESRGAN_Upscaler/
├── input_wallpapers/            # Input folder for raw images (.jpg, .png, .webp)
├── output_wallpapers/           # Output directory for upscaled results
│   ├── desktop_horizontal/      # Automatically categorized landscape wallpapers (>= 16:9)
│   └── mobile_vertical/         # Automatically categorized portrait wallpapers (9:16)
├── models/                      # Offline cache for neural network weights (.pth)
├── anime_upscaler.py            # Automated batch CLI upscaler (PyTorch + CUDA)
├── app_wallpaper_upscaler.py    # Interactive Gradio studio with comparison slider
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Key Modules Explained

- **[`anime_upscaler.py`](anime_upscaler.py):** The headless batch processing engine. Scans `input_wallpapers/`, determines image orientation, runs tiled inference via GPU FP16, and writes lossless `.png` files into organized folders.
- **[`app_wallpaper_upscaler.py`](app_wallpaper_upscaler.py):** A browser-based GUI built on Gradio. Allows drag-and-drop uploading, interactive before/after image comparison sliders, resolution target presets, and aesthetic background padding modes (e.g., blurred background padding for mismatched aspect ratios).
- **[`models/`](models/):** Directory where model checkpoints are cached. Downloaded once automatically via Hugging Face Hub, allowing completely air-gapped/offline operation thereafter.

---

## 🚀 Installation & Setup

### Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python:** Version 3.10, 3.11, or 3.12.
- **GPU (Recommended):** NVIDIA RTX/GTX GPU with CUDA support (4 GB+ VRAM). _Runs on CPU if no CUDA GPU is detected._

### 1. Clone or Open the Workspace

Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:

```bash
# Clone the repository
git clone https://github.com/your-username/RealESRGAN_Upscaler.git
cd RealESRGAN_Upscaler

# Or if downloaded as a ZIP, navigate into the extracted folder:
cd path/to/RealESRGAN_Upscaler
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Windows (CMD):
.venv\Scripts\activate.bat

# Activate on Windows (Git Bash):
source .venv/Scripts/activate

# Activate on Linux/macOS:
source .venv/bin/activate
```

_(On Windows PowerShell, if script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)_

### 3. Install PyTorch with CUDA Support

> [!IMPORTANT]
> **For NVIDIA GeForce RTX 50-Series (RTX 5060 Ti, 5070, 5080, 5090 - Blackwell Architecture):**
> RTX 50-series GPUs use compute capability **`sm_120`**. To avoid `CUDA error: no kernel image is available for execution on the device`, you must install PyTorch with **CUDA 12.8+ (`cu128`)** support.

Install PyTorch with CUDA 12.8:

```bash
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify that CUDA and your architecture are properly recognized:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| Capability:', torch.cuda.get_device_capability(0))"
```

### 4. Install Project Dependencies

Install the remaining packages (Spandrel, Pillow, NumPy, Gradio, Hugging Face Hub):

```bash
pip install -r requirements.txt
```

---

## 🎮 Usage Guide

### Mode 1: Automated Batch Processing (`anime_upscaler.py`)

Best for processing folders of wallpapers in one go.

1. Drop your raw images (`.jpg`, `.jpeg`, `.png`, `.webp`) into `input_wallpapers/`.
2. Execute the script:
   ```bash
   python anime_upscaler.py
   ```
3. The pipeline will:
   - Check if weights exist locally (and auto-download them if missing).
   - Initialize CUDA in FP16 mode.
   - Detect orientation per image (Landscape vs. Portrait).
   - Tile and upscale each image by 4x.
   - Save lossless PNG outputs into `output_wallpapers/desktop_horizontal/` or `output_wallpapers/mobile_vertical/`.

---

### Mode 2: Interactive Web Studio (`app_wallpaper_upscaler.py`)

Best for single-image upscaling, fine-tuning, and direct comparison.

1. Launch the web application:
   ```bash
   python app_wallpaper_upscaler.py
   ```
2. Open the local address in your web browser:
   ```text
   http://127.0.0.1:7860
   ```
3. **Features in the UI:**
   - **Interactive Comparison Slider:** Drag across the image to evaluate line art clarity, sharpness, and artifact removal.
   - **Target Display Presets:**
     - _4K UHD Desktop:_ `3840 × 2160` (16:9)
     - _1440p WQHD Desktop:_ `2560 × 1440` (16:9)
     - _Smartphone Portrait:_ `1080 × 2340` (19.5:9)
     - _Original 4x Upscale:_ Maintains raw aspect ratio at full 4x resolution.
   - **Composition Modes:**
     - _Center Crop:_ Fills the selected target display edge-to-edge.
     - _Blurred Background Pad:_ Keeps the original artwork uncropped and fills pillarbox/letterbox areas with a complementary blurred background.

---

## 🔧 Configuration & Hardware Tuning

Both `anime_upscaler.py` and `app_wallpaper_upscaler.py` expose clear configuration constants at the top of their respective source files:

| Variable                     | Default | Recommended Setting                                                      | Description                                                                                                                                                         |
| :--------------------------- | :------ | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `USE_FP16`                   | `True`  | `True` (NVIDIA GPUs)                                                     | Enables half-precision (Float16) computation on Tensor Cores. Cuts VRAM consumption in half and increases throughput by up to 2x. Set to `False` for CPU execution. |
| `TILE_SIZE`                  | `1024`  | `512` (for 4-8 GB VRAM)`1024` (for 8-16 GB VRAM)`2048` (for 16+ GB VRAM) | Dimension (in pixels) of square image tiles passed to the GPU during inference. Smaller values save VRAM; larger values maximize GPU saturation.                    |
| `TILE_OVERLAP`               | `32`    | `32`                                                                     | Overlap margin in pixels between adjacent tiles to prevent visible border seams during reconstruction.                                                              |
| `FORMAT_TO_EXACT_RESOLUTION` | `False` | User preference                                                          | When set to `True`, crops and scales the final 4x output directly to your primary target monitor resolution using Lanczos resampling.                               |

---

## 🛠️ Troubleshooting & FAQ

### 1. `RuntimeError: CUDA error: no kernel image is available for execution on the device` (or `sm_120 is not compatible`)

- **Cause:** You are running an **NVIDIA GeForce RTX 50-series (Blackwell)** GPU (e.g., RTX 5060 Ti, 5070, 5080, 5090) with compute capability `sm_120`, but your current PyTorch wheel was compiled for CUDA 12.4 or older (which only supports up to `sm_90`).
- **Fix:** Upgrade your PyTorch installation in your virtual environment to CUDA 12.8:
  ```bash
  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

### 2. `torch.cuda.is_available()` returns `False`

- **Cause:** The default CPU-only PyTorch build was installed from standard PyPI.
- **Fix:** Reinstall PyTorch pointing directly to the official CUDA wheels repository:
  ```bash
  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

### 2. CUDA Out-Of-Memory (OOM) Errors

- **Cause:** `TILE_SIZE` is set too high for your graphics card's available VRAM.
- **Fix:** Open `anime_upscaler.py` or `app_wallpaper_upscaler.py` and decrease `TILE_SIZE`:
  ```python
  TILE_SIZE = 512  # Or 256 for systems with <= 4 GB VRAM
  ```

### 3. Model Weights Download Failure

- **Cause:** Restricted network connection or firewall blocking Hugging Face.
- **Manual Fix:** Download [`RealESRGAN_x4plus_anime_6B.pth`](https://huggingface.co/amd/realesrgan-x4plus-anime-6b/resolve/main/RealESRGAN_x4plus_anime_6B.pth) manually and place it directly into the `models/` directory:

  ```text
  RealESRGAN_Upscaler/models/RealESRGAN_x4plus_anime_6B.pth
  ```

  Once stored in `models/`, the application operates with 100% offline autonomy.

---

## 📜 Credits & References

- **Real-ESRGAN:** Wang et al., _Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data_ ([arXiv:2107.10833](https://arxiv.org/abs/2107.10833))
- **Architecture & Weight Loading:** [Spandrel](https://github.com/chaiNNer-org/spandrel) by ChaiNNer
- **Model Checkpoints:** [Hugging Face Hub (`amd/realesrgan-x4plus-anime-6b`)](https://huggingface.co/amd/realesrgan-x4plus-anime-6b)
