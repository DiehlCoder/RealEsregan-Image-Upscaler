import math
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from spandrel import ImageModelDescriptor, ModelLoader

# =====================================================================
# 1. CONFIGURATION & GPU SETTINGS (NVIDIA GeForce RTX 5060 Ti)
# =====================================================================

# Folder paths for inputs, outputs, and local AI model storage
INPUT_DIR = Path("./input_wallpapers")
OUTPUT_DIR = Path("./output_wallpapers")
MODEL_DIR = Path("./models")

# Hugging Face Model: Real-ESRGAN Anime 6B (Runs 100% offline once stored in ./models/)
HF_REPO_ID = "amd/realesrgan-x4plus-anime-6b"
MODEL_FILENAME = "RealESRGAN_x4plus_anime_6B.pth"

# Hardware Acceleration & VRAM Settings (Optimized for NVIDIA RTX 5060 Ti 16 GB)
USE_FP16 = True       # True = Tensor Cores active (2x speed, half VRAM usage)
TILE_SIZE = 1024      # Size of image chunks sent to GPU (1024 is optimal for 16 GB VRAM)
TILE_OVERLAP = 32     # Pixel overlap between tiles to prevent visible seams

# Optional: Format directly to exact monitor or phone resolution (Center Crop + Lanczos)
FORMAT_TO_EXACT_RESOLUTION = False
DESKTOP_TARGET_RES = (3840, 2160)  # 4K UHD (16:9 PC Wallpaper)
MOBILE_TARGET_RES = (1440, 2560)   # WQHD (9:16 Smartphone Wallpaper)

# Supported input file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# =====================================================================
# 2. SETUP & MODEL LOADING
# =====================================================================

def setup_directories() -> tuple[Path, Path]:
    """Creates input and output directories if they do not exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    desktop_out = OUTPUT_DIR / "desktop_horizontal"
    mobile_out = OUTPUT_DIR / "mobile_vertical"

    desktop_out.mkdir(parents=True, exist_ok=True)
    mobile_out.mkdir(parents=True, exist_ok=True)

    return desktop_out, mobile_out


def get_local_model_path() -> Path:
    """
    Checks if model weights already exist in ./models/.
    If missing, downloads them once from Hugging Face for offline use.
    """
    local_file = MODEL_DIR / MODEL_FILENAME
    if local_file.exists():
        return local_file

    print(f"[INFO] Downloading '{MODEL_FILENAME}' from Hugging Face ({HF_REPO_ID})...")
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR,
    )
    print(f"[INFO] Model saved locally to: {downloaded_path}")
    return Path(downloaded_path)


def get_optimal_device() -> torch.device:
    """
    Detects whether CUDA is available and verifies architecture compatibility
    (e.g., checking for Blackwell sm_120 on RTX 50-series GPUs like RTX 5060 Ti).
    """
    if not torch.cuda.is_available():
        print("[WARNUNG] Keine CUDA-GPU gefunden! CPU-Modus aktiv.")
        return torch.device("cpu")

    gpu_name = torch.cuda.get_device_name(0)
    try:
        # Micro-test tensor operation on CUDA to verify kernel availability
        test = torch.zeros((1, 1), device="cuda")
        if USE_FP16:
            test = test.half()
        del test
        torch.cuda.empty_cache()

        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Aktive GPU: {gpu_name} ({total_vram:.1f} GB VRAM, {arch})")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    except Exception as err:
        print("\n" + "=" * 75)
        print(f"⚠️  GPU-INKOMPATIBILITÄT ERKANNT: {gpu_name}")
        print(f"Details: {err}")
        print("\nDeine GPU (RTX 50-Serie Blackwell, sm_120) erfordert PyTorch mit CUDA 12.8+.")
        print("Führe folgenden Befehl im Terminal aus, um die GPU-Beschleunigung zu aktivieren:")
        print("  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("=" * 75 + "\n")
        print("[WARNUNG] Wechsle temporär in den CPU-Modus, um Absturz zu verhindern...\n")
        return torch.device("cpu")


def load_esrgan_model(model_path: Path, device: torch.device) -> tuple[ImageModelDescriptor, torch.device]:
    """Loads the model onto the NVIDIA GPU with FP16 evaluation mode."""
    print(f"[INFO] Loading Real-ESRGAN model onto device: {device}")
    model = ModelLoader().load_from_file(str(model_path))
    assert isinstance(model, ImageModelDescriptor), "Invalid model architecture!"

    try:
        model.to(device).eval()
        if USE_FP16 and device.type == "cuda":
            model.half()
            print("[INFO] Tensor Cores enabled: FP16 (Half-Precision) active for maximum speed.")
    except Exception as err:
        print(f"[WARNUNG] Modell konnte nicht auf {device} geladen werden ({err}). Wechsle zu CPU.")
        device = torch.device("cpu")
        model.to(device).float().eval()

    return model, device


# =====================================================================
# 3. TENSOR TRANSFORMATION & TILED INFERENCE
# =====================================================================

def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Converts a PIL RGB Image into a normalized PyTorch tensor [0.0, 1.0]."""
    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    if USE_FP16 and device.type == "cuda":
        tensor = tensor.half()

    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a PyTorch tensor back into a standard 8-bit RGB PIL Image."""
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
    """
    Splits image into overlapping tiles, upscales on the GPU,
    and stitches them back together seamlessly. Prevents any VRAM overflow.
    """
    scale = model.scale
    batch, channels, height, width = input_tensor.shape

    # Process all at once if within tile limits
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

            # Add overlap padding
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
# 4. MAIN BATCH PIPELINE
# =====================================================================

def process_wallpapers() -> None:
    """Main batch processing function."""
    desktop_out_dir, mobile_out_dir = setup_directories()

    image_files = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"\n[INFO] Keine Bilder in '{INPUT_DIR.resolve()}' gefunden.")
        print("Lege Anime-Bilder in diesen Ordner und starte das Skript erneut!")
        return

    device = get_optimal_device()
    model_path = get_local_model_path()
    model, device = load_esrgan_model(model_path, device)

    print(f"\n[INFO] Starte Batch-Upscaling für {len(image_files)} Wallpaper...\n")

    for index, img_path in enumerate(image_files, start=1):
        start_time = time.perf_counter()

        with Image.open(img_path) as img:
            orig_w, orig_h = img.size

            # Automatisches Erkennen: Desktop (Querformat) vs. Mobile (Hochformat)
            if orig_w >= orig_h:
                target_folder = desktop_out_dir
                target_res = DESKTOP_TARGET_RES
                label = "Desktop (Querformat)"
            else:
                target_folder = mobile_out_dir
                target_res = MOBILE_TARGET_RES
                label = "Mobile (Hochformat)"

            print(f"[{index}/{len(image_files)}] Verarbeite '{img_path.name}' ({orig_w}x{orig_h}) -> {label}")

            # PyTorch Inferenz auf der RTX GPU
            input_tensor = pil_to_tensor(img, device)
            upscaled_tensor = upscale_with_tiling(
                model=model,
                input_tensor=input_tensor,
                tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP
            )
            upscaled_img = tensor_to_pil(upscaled_tensor)

        # Optionales Zuschneiden auf exakte Monitorauflösung
        if FORMAT_TO_EXACT_RESOLUTION:
            upscaled_img = ImageOps.fit(
                upscaled_img,
                target_res,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )

        # Verlustfreie Speicherung als PNG
        out_name = f"{img_path.stem}_upscaled_x{model.scale}.png"
        out_path = target_folder / out_name
        upscaled_img.save(out_path, format="PNG", optimize=True)

        elapsed = time.perf_counter() - start_time
        final_w, final_h = upscaled_img.size
        print(f"    -> Fertig in {elapsed:.2f}s | Auflösung: {final_w}x{final_h} | Gespeichert: {out_path.name}\n")

    print("[ERFOLGREICH] Alle Anime-Wallpaper wurden lokal mit der GPU hochskaliert!")


if __name__ == "__main__":
    process_wallpapers()
