import math
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
from spandrel import ImageModelDescriptor, ModelLoader

# =====================================================================
# 1. CONFIGURATION & ADJUSTABLE PARAMETERS
# =====================================================================

MODEL_DIR = Path("./models")
OUTPUT_DIR = Path("./output_wallpapers")

HF_REPO_ID = "amd/realesrgan-x4plus-anime-6b"
MODEL_FILENAME = "RealESRGAN_x4plus_anime_6B.pth"

# Hardware Acceleration Settings
USE_FP16 = True
TILE_SIZE = 1024
TILE_OVERLAP = 32

# Target Resolutions
RES_MONITOR_L32P30 = (3840, 2160)   # 4K UHD (16:9)
RES_MONITOR_AG322 = (2560, 1440)    # 1440p QHD (16:9)
RES_GALAXY_S24 = (1080, 2340)       # Mobile Portrait (19.5:9)

SERVER_PORT = 7860
AUTO_OPEN_BROWSER = True


# =====================================================================
# 2. AI MODEL INITIALIZATION (PYTORCH + SPANDREL)
# =====================================================================

def get_optimal_device() -> torch.device:
    """Detects whether CUDA is available and verifies kernel compatibility."""
    if not torch.cuda.is_available():
        print("[INFO] No CUDA GPU detected. Running in CPU mode.")
        return torch.device("cpu")

    gpu_name = torch.cuda.get_device_name(0)
    try:
        test = torch.zeros((1, 1), device="cuda")
        if USE_FP16:
            test = test.half()
        del test
        torch.cuda.empty_cache()

        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        print(f"[INFO] CUDA GPU ready: {gpu_name} ({arch})")
        return torch.device("cuda")
    except Exception as err:
        print("\n" + "=" * 75)
        print(f"GPU INCOMPATIBILITY DETECTED: {gpu_name}")
        print(f"Details: {err}")
        print("\nYour GPU requires PyTorch with CUDA 12.8+.")
        print("Run the following command to enable full GPU acceleration:")
        print("  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("=" * 75 + "\n")
        print("[WARNING] Temporarily falling back to CPU mode...\n")
        return torch.device("cpu")


def init_environment_and_model() -> tuple[ImageModelDescriptor, torch.device]:
    """Downloads model weights if missing and initializes on target device."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    local_model_path = MODEL_DIR / MODEL_FILENAME
    if not local_model_path.exists():
        print(f"[INFO] Downloading '{MODEL_FILENAME}' from Hugging Face...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODEL_DIR,
        )

    device = get_optimal_device()
    print(f"[INFO] Loading AI Model on device: {device}")

    model = ModelLoader().load_from_file(str(local_model_path))
    assert isinstance(model, ImageModelDescriptor), "Invalid model format!"

    try:
        model.to(device).eval()
        if USE_FP16 and device.type == "cuda":
            model.half()
            print("[INFO] Tensor Cores enabled: FP16 active for maximum speed.")
    except Exception as err:
        print(f"[WARNING] Model placement on {device} failed ({err}). Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device).float().eval()

    return model, device


# Load the model globally on startup so it stays ready in VRAM
AI_MODEL, DEVICE = init_environment_and_model()


# =====================================================================
# 3. TENSOR CONVERSION & TILED GPU INFERENCE
# =====================================================================

def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Converts a PIL RGB image into a normalized PyTorch GPU tensor."""
    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)
    if USE_FP16 and device.type == "cuda":
        tensor = tensor.half()
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a PyTorch GPU tensor back into a standard 8-bit RGB PIL image."""
    output = tensor.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    output_uint8 = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output_uint8)


@torch.inference_mode()
def upscale_with_tiling(
    model: ImageModelDescriptor,
    input_tensor: torch.Tensor,
    tile_size: int = 1024,
    tile_overlap: int = 32
) -> torch.Tensor:
    """Splits image into overlapping tiles, upscales on GPU, and stitches seamlessly."""
    scale = model.scale
    batch, channels, height, width = input_tensor.shape

    if height <= tile_size and width <= tile_size:
        return model(input_tensor)

    out_height = height * scale
    out_width = width * scale
    output_tensor = torch.zeros(
        (batch, channels, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            in_x1 = ofs_x
            in_x2 = min(ofs_x + tile_size, width)
            in_y1 = ofs_y
            in_y2 = min(ofs_y + tile_size, height)

            pad_x1 = max(in_x1 - tile_overlap, 0)
            pad_x2 = min(in_x2 + tile_overlap, width)
            pad_y1 = max(in_y1 - tile_overlap, 0)
            pad_y2 = min(in_y2 + tile_overlap, height)

            in_tile = input_tensor[:, :, pad_y1:pad_y2, pad_x1:pad_x2]
            out_tile = model(in_tile)

            out_x1 = in_x1 * scale
            out_x2 = in_x2 * scale
            out_y1 = in_y1 * scale
            out_y2 = in_y2 * scale

            tile_x1 = (in_x1 - pad_x1) * scale
            tile_x2 = tile_x1 + (in_x2 - in_x1) * scale
            tile_y1 = (in_y1 - pad_y1) * scale
            tile_y2 = tile_y1 + (in_y2 - in_y1) * scale

            output_tensor[:, :, out_y1:out_y2, out_x1:out_x2] = out_tile[
                :, :, tile_y1:tile_y2, tile_x1:tile_x2
            ]

    return output_tensor


# =====================================================================
# 4. WALLPAPER FORMATTING & ASPECT RATIO FUNCTIONS
# =====================================================================

def format_to_target_screen(
    img: Image.Image,
    target_res: tuple[int, int],
    is_crop: bool,
    resample_method: Image.Resampling
) -> Image.Image:
    """Formats an image to the exact device resolution via Crop or Blurred Background."""
    target_w, target_h = target_res

    if is_crop:
        return ImageOps.fit(
            img,
            (target_w, target_h),
            method=resample_method,
            centering=(0.5, 0.5)
        )
    else:
        background = ImageOps.fit(
            img,
            (target_w, target_h),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5)
        ).filter(ImageFilter.GaussianBlur(radius=45))

        foreground = ImageOps.contain(img, (target_w, target_h), method=resample_method)

        paste_x = (target_w - foreground.width) // 2
        paste_y = (target_h - foreground.height) // 2
        background.paste(foreground, (paste_x, paste_y))
        return background


# =====================================================================
# 5. MAIN PROCESSING PIPELINE
# =====================================================================

def process_single_wallpaper(
    input_img: Image.Image,
    device_preset: str,
    fit_mode: str
) -> tuple[tuple[Image.Image, Image.Image], str, str]:
    """Runs AI inference and returns comparison images, download path, and HUD metrics."""
    if input_img is None:
        raise gr.Error("Please upload an image first.")

    start_time = time.perf_counter()
    orig_w, orig_h = input_img.size
    is_horizontal = orig_w >= orig_h

    # Parse target resolution cleanly
    preset_lower = device_preset.lower()
    if "auto" in preset_lower:
        if is_horizontal:
            target_res = RES_MONITOR_L32P30
            subfolder_name = "Desktop_4K_3840x2160"
            device_label = "Desktop 4K UHD"
        else:
            target_res = RES_GALAXY_S24
            subfolder_name = "Mobile_1080x2340"
            device_label = "Mobile Portrait"
    elif "4k" in preset_lower or "3840" in preset_lower:
        target_res = RES_MONITOR_L32P30
        subfolder_name = "Desktop_4K_3840x2160"
        device_label = "Desktop 4K UHD"
    elif "1440" in preset_lower or "2560" in preset_lower:
        target_res = RES_MONITOR_AG322
        subfolder_name = "Desktop_1440p_2560x1440"
        device_label = "Desktop 1440p QHD"
    elif "phone" in preset_lower or "mobile" in preset_lower or "2340" in preset_lower:
        target_res = RES_GALAXY_S24
        subfolder_name = "Mobile_1080x2340"
        device_label = "Mobile Portrait"
    else:
        target_res = None
        subfolder_name = "Native_4x_Upscale"
        device_label = "Native 4x AI"

    is_crop = ("crop" in fit_mode.lower() or "fill" in fit_mode.lower())

    # Step 2: 4x Super-Resolution
    input_tensor = pil_to_tensor(input_img, DEVICE)
    upscaled_tensor = upscale_with_tiling(
        model=AI_MODEL,
        input_tensor=input_tensor,
        tile_size=TILE_SIZE,
        tile_overlap=TILE_OVERLAP
    )
    upscaled_4x_img = tensor_to_pil(upscaled_tensor)
    raw_4x_w, raw_4x_h = upscaled_4x_img.size

    # Step 3: Device Resizing for Comparison
    if target_res is not None:
        final_after_img = format_to_target_screen(
            img=upscaled_4x_img,
            target_res=target_res,
            is_crop=is_crop,
            resample_method=Image.Resampling.LANCZOS
        )
        before_comparison_img = format_to_target_screen(
            img=input_img.convert("RGB"),
            target_res=target_res,
            is_crop=is_crop,
            resample_method=Image.Resampling.NEAREST
        )
    else:
        final_after_img = upscaled_4x_img
        before_comparison_img = input_img.convert("RGB").resize(
            (raw_4x_w, raw_4x_h),
            resample=Image.Resampling.NEAREST
        )

    # Step 4: Save Lossless PNG
    target_folder = OUTPUT_DIR / subfolder_name
    target_folder.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_w, final_h = final_after_img.size
    output_filename = f"upscaled_{final_w}x{final_h}_{timestamp}.png"
    output_path = target_folder / output_filename
    final_after_img.save(output_path, format="PNG", optimize=True)

    elapsed = time.perf_counter() - start_time

    # Return sleek HTML metric cards
    hud_html = f"""
    <div class="hud-metrics-container">
        <div class="hud-metric">
            <span class="hud-tag">TIME</span>
            <span class="hud-stat">{elapsed:.2f}s</span>
        </div>
        <div class="hud-metric">
            <span class="hud-tag">SOURCE</span>
            <span class="hud-stat">{orig_w} × {orig_h}</span>
        </div>
        <div class="hud-metric">
            <span class="hud-tag">TARGET</span>
            <span class="hud-stat accent">{final_w} × {final_h}</span>
        </div>
        <div class="hud-metric">
            <span class="hud-tag">PROFILE</span>
            <span class="hud-stat">{device_label}</span>
        </div>
    </div>
    """

    return (before_comparison_img, final_after_img), str(output_path), hud_html


# =====================================================================
# 6. UI STYLING & ASSETS (3-COLOR PALETTE & SPACE GROTESK)
# =====================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* =====================================================================
   THREE-COLOR DESIGN TOKENS
   Dark Mode:  Base #090d16 | Surface #111827 | Accent #00f0ff
   Light Mode: Base #f8fafc | Surface #ffffff | Accent #0284c7
   ===================================================================== */

:root {
    --bg-base: #f8fafc;
    --bg-surface: #ffffff;
    --accent: #0284c7;
    --accent-glow: rgba(2, 132, 199, 0.12);
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --border-subtle: rgba(2, 132, 199, 0.16);
    --border-card: #e2e8f0;
}

.dark, body.dark {
    --bg-base: #090d16;
    --bg-surface: #111827;
    --accent: #00f0ff;
    --accent-glow: rgba(0, 240, 255, 0.22);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(0, 240, 255, 0.22);
    --border-card: rgba(255, 255, 255, 0.08);
}

* {
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* Header Navbar */
.studio-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}

.dark .studio-navbar {
    box-shadow: 0 0 25px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.brand-section {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    color: var(--accent);
}

.dark .brand-icon {
    box-shadow: 0 0 14px var(--accent-glow);
}

.brand-title {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--text-primary);
}

.brand-title span {
    color: var(--accent);
}

.brand-subtitle {
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 0.02em;
}

.nav-badges {
    display: flex;
    align-items: center;
    gap: 12px;
}

.gpu-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--accent);
}

.dark .gpu-badge {
    box-shadow: 0 0 10px var(--accent-glow);
}

.pulse-circle {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--accent);
}

/* Theme Toggle Button */
.theme-switch-btn {
    background: var(--bg-base) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    padding: 8px 14px !important;
    height: auto !important;
}

.theme-switch-btn:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Control Panel & Containers */
.control-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03) !important;
}

.dark .control-card {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

/* Primary Action Button (Gaming HUD) */
.primary-upscale-btn {
    background: var(--accent) !important;
    color: #050811 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 14px 20px !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    margin-top: 8px !important;
}

.primary-upscale-btn:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.08) !important;
    box-shadow: 0 6px 22px var(--accent) !important;
}

/* HUD Metrics Grid */
.hud-metrics-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-top: 10px;
}

.hud-metric {
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
}

.hud-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
}

.hud-stat {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}

.hud-stat.accent {
    color: var(--accent);
}

/* Comparison Preview Container */
.preview-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    display: flex !important;
    flex-direction: column !important;
}

.dark .preview-card {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

/* Gradio Component Cleanups */
.gr-form, .gr-box {
    background: transparent !important;
    border: none !important;
}

label span {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}

footer {
    display: none !important;
}
"""

INIT_THEME_JS = """
() => {
    const saved = localStorage.getItem('agy_theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = saved === 'dark' || (!saved && prefersDark) || true;
    if (isDark) {
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    }
}
"""

TOGGLE_THEME_JS = """
() => {
    const isDark = document.documentElement.classList.toggle('dark');
    document.body.classList.toggle('dark', isDark);
    localStorage.setItem('agy_theme', isDark ? 'dark' : 'light');
}
"""


# =====================================================================
# 7. INTERACTIVE WEB APPLICATION UI (GRADIO BLOCKS)
# =====================================================================

def build_web_app() -> gr.Blocks:
    """Builds the sleek, game-inspired Real-ESRGAN Studio interface."""
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Space Grotesk"),
    )

    with gr.Blocks(title="Real-ESRGAN Studio", theme=custom_theme, css=CUSTOM_CSS) as app:
        
        # Sleek Minimalist Navbar Header
        with gr.Row(elem_classes=["studio-navbar"]):
            with gr.Column(scale=9):
                gr.HTML(
                    """
                    <div class="brand-section">
                        <div class="brand-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                                <polyline points="2 17 12 22 22 17"></polyline>
                                <polyline points="2 12 12 17 22 12"></polyline>
                            </svg>
                        </div>
                        <div>
                            <div class="brand-title">APEX UPSCALE <span>STUDIO</span></div>
                            <div class="brand-subtitle">Real-ESRGAN 4x Anime Engine & Aspect Ratio Formatter</div>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=3, min_width=180):
                with gr.Row():
                    gr.HTML(
                        """
                        <div class="nav-badges">
                            <div class="gpu-badge">
                                <span class="pulse-circle"></span>
                                <span>CUDA ACTIVE</span>
                            </div>
                        </div>
                        """
                    )
                    theme_btn = gr.Button("Mode", elem_classes=["theme-switch-btn"])

        # Main Workspace: 2-Column Balanced Layout
        with gr.Row():
            
            # Left Column: Streamlined Controls
            with gr.Column(scale=4, min_width=360, elem_classes=["control-card"]):
                
                input_image = gr.Image(
                    type="pil",
                    label="Source Image",
                    height=240,
                )

                device_dropdown = gr.Dropdown(
                    choices=[
                        "Auto Detect (Desktop 4K / Mobile WQHD)",
                        "Desktop 4K UHD (3840 × 2160)",
                        "Desktop 1440p QHD (2560 × 1440)",
                        "Mobile Portrait (1080 × 2340)",
                        "Native 4x Upscale (No Crop)",
                    ],
                    value="Auto Detect (Desktop 4K / Mobile WQHD)",
                    label="Target Display Preset"
                )

                fit_mode_radio = gr.Radio(
                    choices=[
                        "Fill Screen (Crop)",
                        "Keep Artwork (Blurred Border)",
                    ],
                    value="Fill Screen (Crop)",
                    label="Framing Mode"
                )

                upscale_btn = gr.Button(
                    "Enhance Wallpaper",
                    variant="primary",
                    elem_classes=["primary-upscale-btn"]
                )

                metrics_output = gr.HTML(
                    """
                    <div class="hud-metrics-container">
                        <div class="hud-metric">
                            <span class="hud-tag">STATUS</span>
                            <span class="hud-stat">STANDBY</span>
                        </div>
                        <div class="hud-metric">
                            <span class="hud-tag">ENGINE</span>
                            <span class="hud-stat accent">ESRGAN 6B</span>
                        </div>
                    </div>
                    """
                )

                download_file = gr.File(
                    label="Download Lossless PNG",
                    interactive=False
                )

            # Right Column: Interactive Studio Comparison Viewport
            with gr.Column(scale=7, elem_classes=["preview-card"]):
                
                gr.HTML(
                    """
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                                <line x1="8" y1="21" x2="16" y2="21"></line>
                                <line x1="12" y1="17" x2="12" y2="21"></line>
                            </svg>
                            <span>Comparison Studio</span>
                        </div>
                        <div style="font-size: 12px; font-weight: 500; opacity: 0.75;">
                            <span>Left: Original</span> &nbsp;|&nbsp; <span style="color: var(--accent); font-weight: 700;">Right: 4x Enhanced</span>
                        </div>
                    </div>
                    """
                )

                comparison_slider = gr.ImageSlider(
                    show_label=False,
                    type="pil",
                    format="png",
                )

        # Wire Up Event Listeners
        upscale_btn.click(
            fn=process_single_wallpaper,
            inputs=[input_image, device_dropdown, fit_mode_radio],
            outputs=[comparison_slider, download_file, metrics_output]
        )

        theme_btn.click(
            None,
            None,
            None,
            js=TOGGLE_THEME_JS
        )

        app.load(
            None,
            None,
            None,
            js=INIT_THEME_JS
        )

    return app


if __name__ == "__main__":
    web_app = build_web_app()
    web_app.launch(server_port=SERVER_PORT, inbrowser=AUTO_OPEN_BROWSER)
