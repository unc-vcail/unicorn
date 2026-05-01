import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm
from PIL import Image, ImageChops, ImageEnhance, ImageOps

from hints.refineblur_gpu import get_blur_hints_gpu
from hints.refinehaze_gpu import get_haze_hints_gpu

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    GPU_AVAILABLE = True
    print("CuPy GPU acceleration available")
except ImportError:
    print("CuPy not available, falling back to CPU")
    GPU_AVAILABLE = False
    cp = np

try:
    TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False

from skimage import filters, feature

import open_clip1


# ── GPU/CPU transfer helpers ──────────────────────────────────────────────────

def to_gpu(array):
    if GPU_AVAILABLE:
        return cp.asarray(array)
    return array

def to_cpu(array):
    if GPU_AVAILABLE and hasattr(array, 'get'):
        return array.get()
    return array


# ── GPU-accelerated primitive operations ─────────────────────────────────────

def gpu_gaussian_filter(image, sigma, order=0, axis=None):
    if GPU_AVAILABLE:
        gpu_img = to_gpu(image)
        if axis is not None:
            order_array = [0] * len(image.shape)
            order_array[axis] = order
            result = cp_ndimage.gaussian_filter(gpu_img, sigma, order=order_array)
        else:
            result = cp_ndimage.gaussian_filter(gpu_img, sigma)
        return to_cpu(result)
    else:
        if axis is not None:
            order_array = [0] * len(image.shape)
            order_array[axis] = order
            return ndimage.gaussian_filter(image, sigma, order=order_array)
        return ndimage.gaussian_filter(image, sigma)

def gpu_sobel_filter(image, axis):
    if GPU_AVAILABLE:
        return to_cpu(cp_ndimage.sobel(to_gpu(image), axis=axis))
    return ndimage.sobel(image, axis=axis)

def gpu_convolve(image, kernel, mode='wrap'):
    if GPU_AVAILABLE:
        return to_cpu(cp_ndimage.convolve(to_gpu(image), to_gpu(kernel), mode=mode))
    elif TORCH_AVAILABLE:
        device = torch.device('cuda')
        img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        ker_t = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(img_t, ker_t, padding='same').squeeze().cpu().numpy()
    return ndimage.convolve(image, kernel, mode=mode)

def gpu_fft2(image):
    if GPU_AVAILABLE:
        return cp.fft.fft2(to_gpu(image))  # stays on GPU for chained ops
    return np.fft.fft2(image)

def gpu_ifft2(freq_image):
    if GPU_AVAILABLE:
        return to_cpu(cp.fft.ifft2(freq_image))
    return np.fft.ifft2(freq_image)


# ── Low-level cue extractors ──────────────────────────────────────────────────

def extract_structure_tensor_coherence_gpu(image, sigma=1.0, rho=3.0):
    """Structure Tensor Coherence (Bigun et al. 1991): measures local orientation strength."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = gray.astype(np.float64)

    grad_x = gpu_gaussian_filter(gray, sigma, order=1, axis=1)
    grad_y = gpu_gaussian_filter(gray, sigma, order=1, axis=0)

    if GPU_AVAILABLE:
        gx, gy = to_gpu(grad_x), to_gpu(grad_y)
        J11 = to_cpu(cp_gaussian_filter(gx * gx, rho))
        J12 = to_cpu(cp_gaussian_filter(gx * gy, rho))
        J22 = to_cpu(cp_gaussian_filter(gy * gy, rho))
    else:
        J11 = ndimage.gaussian_filter(grad_x * grad_x, rho)
        J12 = ndimage.gaussian_filter(grad_x * grad_y, rho)
        J22 = ndimage.gaussian_filter(grad_y * grad_y, rho)

    trace = J11 + J22
    det = J11 * J22 - J12 * J12
    lam1 = 0.5 * (trace + np.sqrt(np.maximum(trace**2 - 4*det, 0) + 1e-10))
    lam2 = 0.5 * (trace - np.sqrt(np.maximum(trace**2 - 4*det, 0) + 1e-10))
    return (lam1 - lam2) / (lam1 + lam2 + 1e-10)


def extract_bright_channel_prior_gpu(image, patch_size=15, use_guided_filter=True,
                                     guide_radius=8, guide_eps=0.01):
    """Bright Channel Prior (Tan 2008): proxy for haze/atmospheric scattering density."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image.astype(np.float64) / 255.0
    h, w, c = image.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    if GPU_AVAILABLE:
        bright = cp.zeros((h, w), dtype=cp.float64)
        gpu_img = to_gpu(image)
        for i in range(c):
            bright = cp.maximum(bright, to_gpu(cv2.dilate(to_cpu(gpu_img[:, :, i]), kernel)))
        bright = to_cpu(bright)
    else:
        bright = np.zeros((h, w), dtype=np.float64)
        for i in range(c):
            bright = np.maximum(bright, cv2.dilate(image[:, :, i], kernel))

    if use_guided_filter:
        try:
            from hints.refinehaze_gpu import GuidedFilterGPU
            bright = GuidedFilterGPU(image * 255, guide_radius, guide_eps).filter(bright)
        except ImportError:
            pass

    return bright


def extract_gabor_energy_map_gpu(image, frequencies=[0.1, 0.2, 0.3], orientations=8):
    """Gabor filter bank energy (Daugman 1985): captures texture frequency and orientation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = gray.astype(np.float64)

    if GPU_AVAILABLE:
        responses = []
        gpu_gray = to_gpu(gray)
        for freq in frequencies:
            for i in range(orientations):
                theta = i * np.pi / orientations
                k = filters.gabor_kernel(freq, theta=theta)
                fr = cp_ndimage.convolve(gpu_gray, to_gpu(np.real(k)), mode='wrap')
                fi = cp_ndimage.convolve(gpu_gray, to_gpu(np.imag(k)), mode='wrap')
                responses.append(cp.sqrt(fr**2 + fi**2))
        result = cp.mean(cp.stack(responses, axis=0), axis=0)
        return to_cpu(result)
    else:
        responses = []
        for freq in frequencies:
            for i in range(orientations):
                theta = i * np.pi / orientations
                k = filters.gabor_kernel(freq, theta=theta)
                fr = ndimage.convolve(gray, np.real(k), mode='wrap')
                fi = ndimage.convolve(gray, np.imag(k), mode='wrap')
                responses.append(np.sqrt(fr**2 + fi**2))
        return np.mean(responses, axis=0)


def extract_gradient_magnitude_map_gpu(image, sigma=1.0):
    """Multi-scale gradient magnitude (Lindeberg 1998): highlights edges at 3 scales."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = gray.astype(np.float64)
    scales = [sigma * (2**i) for i in range(3)]

    if GPU_AVAILABLE:
        gpu_gray = to_gpu(gray)
        combined = cp.zeros_like(gpu_gray)
        for s in scales:
            gx = cp_ndimage.gaussian_filter(gpu_gray, s, order=[0, 1])
            gy = cp_ndimage.gaussian_filter(gpu_gray, s, order=[1, 0])
            combined = cp.maximum(combined, cp.sqrt(gx**2 + gy**2) * (s**0.5))
        return to_cpu(combined)
    else:
        combined = np.zeros_like(gray)
        for s in scales:
            gx = ndimage.gaussian_filter(gray, s, order=[0, 1])
            gy = ndimage.gaussian_filter(gray, s, order=[1, 0])
            combined = np.maximum(combined, np.sqrt(gx**2 + gy**2) * (s**0.5))
        return combined


def extract_laplacian_pyramid_residuals_gpu(image, levels=4):
    """Laplacian pyramid residuals (Burt & Adelson 1983): captures multi-scale detail loss."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = gray.astype(np.float64)

    pyr = [gray]
    cur = gray.copy()
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)

    if GPU_AVAILABLE:
        combined = cp.zeros_like(to_gpu(pyr[0]))
        for i in range(len(pyr) - 1):
            expanded = cv2.pyrUp(pyr[i + 1])
            if expanded.shape != pyr[i].shape:
                expanded = cv2.resize(expanded, (pyr[i].shape[1], pyr[i].shape[0]))
            lap = to_gpu(pyr[i]) - to_gpu(expanded)
            if lap.shape != combined.shape:
                lap = to_gpu(cv2.resize(to_cpu(lap), (combined.shape[1], combined.shape[0])))
            combined += cp.abs(lap)
        return to_cpu(combined)
    else:
        combined = np.zeros_like(pyr[0])
        for i in range(len(pyr) - 1):
            expanded = cv2.pyrUp(pyr[i + 1])
            if expanded.shape != pyr[i].shape:
                expanded = cv2.resize(expanded, (pyr[i].shape[1], pyr[i].shape[0]))
            lap = pyr[i] - expanded
            if lap.shape != combined.shape:
                lap = cv2.resize(lap, (combined.shape[1], combined.shape[0]))
            combined += np.abs(lap)
        return combined


def extract_local_standard_deviation_gpu(image, window_size=9):
    """Local standard deviation (Haralick et al. 1973): measures local texture variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = gray.astype(np.float64)

    if GPU_AVAILABLE:
        gpu_gray = to_gpu(gray)
        k = cp.ones((window_size, window_size)) / (window_size**2)
        mu = cp_ndimage.convolve(gpu_gray, k, mode='reflect')
        mu2 = cp_ndimage.convolve(gpu_gray**2, k, mode='reflect')
        return to_cpu(cp.sqrt(cp.maximum(mu2 - mu**2, 0)))
    else:
        k = np.ones((window_size, window_size), np.float32) / (window_size**2)
        mu = cv2.filter2D(gray, -1, k)
        mu2 = cv2.filter2D(gray**2, -1, k)
        return np.sqrt(np.maximum(mu2 - mu**2, 0))


def extract_noise_map_gpu(image):
    """Per-pixel noise estimate: max(|∇x|, |∇y|) across all channels."""
    if len(image.shape) == 3:
        if GPU_AVAILABLE:
            gpu_img = to_gpu(image)
            maps = []
            for c in range(image.shape[2]):
                ch = gpu_img[:, :, c].astype(cp.float64)
                maps.append(cp.maximum(cp.abs(cp_ndimage.sobel(ch, axis=1)),
                                       cp.abs(cp_ndimage.sobel(ch, axis=0))))
            return to_cpu(cp.max(cp.stack(maps, axis=0), axis=0))
        else:
            maps = []
            for c in range(image.shape[2]):
                ch = image[:, :, c].astype(np.float64)
                maps.append(np.maximum(np.abs(ndimage.sobel(ch, axis=1)),
                                       np.abs(ndimage.sobel(ch, axis=0))))
            return np.maximum.reduce(maps)
    else:
        if GPU_AVAILABLE:
            ch = to_gpu(image.astype(np.float64))
            return to_cpu(cp.maximum(cp.abs(cp_ndimage.sobel(ch, axis=1)),
                                     cp.abs(cp_ndimage.sobel(ch, axis=0))))
        else:
            ch = image.astype(np.float64)
            return np.maximum(np.abs(ndimage.sobel(ch, axis=1)),
                              np.abs(ndimage.sobel(ch, axis=0)))


def extract_local_binary_pattern_gpu(image, radius=3, n_points=None):
    """LBP texture descriptor (CPU; scikit-image is already optimised)."""
    if n_points is None:
        n_points = 8 * radius
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    return feature.local_binary_pattern(gray, n_points, radius, method='uniform')


def extract_saturation_map_gpu(image):
    """HSV saturation channel — zero for grayscale inputs."""
    if len(image.shape) == 2:
        return np.zeros_like(image, dtype=np.float64)
    hsv = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].astype(np.float64)


# ── Task-specific hint generators ─────────────────────────────────────────────

def generate_blur_hints_gpu(img):
    """Wiener-deconvolved estimate and shock-filtered edge map for blur guidance."""
    _, x, e = get_blur_hints_gpu(img)
    return [np.clip(x, 0, 1), e]

def generate_haze_hints_gpu(img):
    """Dark-channel prior transmission map for haze guidance."""
    _, t, _ = get_haze_hints_gpu(img)
    return [t]

def generate_lowc_hints(img):
    """Normalised colour map and gradient-edge map for low-light guidance."""
    h2 = np.clip(img / (img.mean(axis=-1)[..., None] + 1e-3) - 0.5, 0, 1)
    ty, tx = np.gradient(h2, axis=[0, 1])
    return [(h2 * 255).astype(np.uint8),
            (np.maximum(np.abs(ty), np.abs(tx)) * 255).astype(np.uint8)]


# ── Full cue extraction pipeline ──────────────────────────────────────────────

def extract_all_cues_gpu_batch(image):
    """
    Extract the complete set of low-level guidance cues for a single image.
    Uses CuPy GPU acceleration where available, falls back to CPU otherwise.
    Returns a dict mapping cue name -> numpy array.
    """
    cues = {}
    cues['original'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cues['Structure Tensor Coherence'] = extract_structure_tensor_coherence_gpu(image)
    cues['Bright Channel Prior'] = extract_bright_channel_prior_gpu(image)
    cues['Local Binary Pattern'] = extract_local_binary_pattern_gpu(image)
    cues['Gabor Energy'] = extract_gabor_energy_map_gpu(image) * 3
    cues['Multi-scale Gradient'] = extract_gradient_magnitude_map_gpu(image)
    cues['Laplacian Residuals'] = extract_laplacian_pyramid_residuals_gpu(image)
    cues['Local Std Deviation'] = extract_local_standard_deviation_gpu(image)
    cues['Saturation Map'] = extract_saturation_map_gpu(image)
    cues['Noise Map'] = extract_noise_map_gpu(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for ix, cue in enumerate(generate_blur_hints_gpu(rgb)):
        cues[f'Shock Map {ix}'] = cue
    for ix, cue in enumerate(generate_haze_hints_gpu(rgb)):
        cues[f'Haze Map {ix}'] = cue
    for ix, cue in enumerate(generate_lowc_hints(rgb)):
        cues[f'Color Map {ix}'] = cue

    return cues


# ── Image I/O ─────────────────────────────────────────────────────────────────

def resize_image_with_torch(image_path, target_size=512):
    """Read and resize an image using PyTorch Lanczos interpolation."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_resized = transforms.Resize(target_size, interpolation=3)(img_tensor)
    img_numpy = (img_resized * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)


# ── Optional degradation classifier (DA-CLIP) ─────────────────────────────────

def check_degra_features(degra_features, daclip_checkpoint):
    """Identify dominant degradation type by matching DA-CLIP features to text labels."""
    tokenizer = open_clip1.get_tokenizer('ViT-B-32')
    degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy',
                    'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted', 'defocus-blurry']
    text = tokenizer(degradations)
    model, _ = open_clip1.create_model_from_pretrained('daclip_ViT-B-32', pretrained=daclip_checkpoint)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
        idx = torch.argmax(probs[0])
    return f"{degradations[idx]} ({probs[0][idx]:.3f})"


# ── Main dataset processor ────────────────────────────────────────────────────

def process_datasets(dataset_configs: List[Union[str, dict]], output_dir: str,
                     target_size: int = 512, generate_features: bool = False,
                     generate_captions: bool = False,
                     daclip_checkpoint: str = './ckpts/daclip_ViT-B-32.pt'):
    """
    Process one or more image datasets, generating low-level cues for each image.

    For every input image the script:
      1. Resizes to `target_size` (shortest side).
      2. Optionally applies a dataset-specific augmentation (e.g. JPEG, noise).
      3. Extracts all low-level cues and writes each as a PNG.
      4. Optionally saves DA-CLIP degradation feature vectors (.pth).
      5. Optionally writes placeholder BLIP captions for the clean target.

    Args:
        dataset_configs: list of path strings or dicts with keys:
            root            -- dataset root used to compute relative output paths
            dataset_root    -- directory containing degraded input images
            augment_fn      -- optional callable(img_np) -> img_np
            name            -- display name for progress bars
            target_path_fn  -- callable(path_str, index) -> target_path_str
        output_dir:         where per-image subdirectories are written
        target_size:        resize shortest side to this before processing
        generate_features:  save DA-CLIP degradation feature vectors
        generate_captions:  write caption placeholder files (fill in as needed)
        daclip_checkpoint:  path to DA-CLIP ViT-B-32 weights
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    daclip_model, daclip_preprocess = None, None
    if generate_features:
        print("Loading DACLIP model...")
        daclip_model, daclip_preprocess = open_clip1.create_model_from_pretrained(
            'daclip_ViT-B-32', pretrained=daclip_checkpoint)
        daclip_model.eval()

    if generate_captions:
        print("Loading CLIP Interrogator models...")
        from clip_interrogator import Config, Interrogator
        cfg = Config(clip_model_name="ViT-L-14/openai", device='cuda', quiet=True,
                     caption_model_name='blip-large')
        blip1_5 = Interrogator(cfg)
        cfg = Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", device='cuda', quiet=True,
                     caption_model_name='blip-large')
        blip2_1 = Interrogator(cfg)

    parsed_configs = []
    total_files = 0

    print("Scanning datasets...")
    for config in tqdm(dataset_configs, desc="Scanning"):
        if isinstance(config, str):
            root = dataset_root = Path(config)
            augment_fn, dataset_name = None, root.name
            target_path_fn = lambda x, ix: x
        else:
            root = Path(config['root'])
            dataset_root = Path(config['dataset_root'])
            augment_fn = config.get('augment_fn', None)
            dataset_name = config.get('name', dataset_root.name)
            target_path_fn = config.get('target_path_fn', lambda x, ix: x)

        if not dataset_root.exists():
            print(f"Warning: {dataset_root} not found, skipping")
            continue

        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_root.glob(f"**/*{ext}"))
            image_files.extend(dataset_root.glob(f"**/*{ext.upper()}"))

        parsed_configs.append({
            'root': root, 'dataset_root': dataset_root, 'augment_fn': augment_fn,
            'name': dataset_name, 'files': sorted(image_files),
            'target_path_fn': target_path_fn,
        })
        total_files += len(image_files)

    print(f"Total files to process: {total_files}")
    overall_pbar = tqdm(total=total_files, desc="Overall Progress", position=0)

    for config in parsed_configs:
        dataset_pbar = tqdm(config['files'], desc=f"Dataset: {config['name']}",
                            position=1, leave=False)
        for ix, img_path in enumerate(dataset_pbar):
            if '.ipynb_checkpoints' in str(img_path):
                continue

            output_subdir = Path(output_dir) / img_path.relative_to(config['root']).parent / img_path.stem
            output_subdir.mkdir(parents=True, exist_ok=True)

            target_img = resize_image_with_torch(config['target_path_fn'](str(img_path), ix), target_size)
            resized_img = resize_image_with_torch(str(img_path), target_size)
            if config['augment_fn'] is not None:
                resized_img = config['augment_fn'](resized_img)

            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            if generate_features:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, feats = daclip_model.encode_image(
                        daclip_preprocess(pil_image).unsqueeze(0), control=True)
                    feats /= feats.norm(dim=-1, keepdim=True)
                    torch.save(feats.cpu(), output_subdir / 'degra_features.pth')

            if generate_captions:
                # Fill in caption generation calls here if needed:
                # caption = image_to_prompt(pil_image, blip1_5, mode='fast')
                caption = ""
                with open(output_subdir / 'caption.txt', 'w') as f:
                    f.write(caption.encode('ascii', errors='ignore').decode('ascii'))

            cues = extract_all_cues_gpu_batch(resized_img)
            cues['Target'] = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

            for name, cue_img in cues.items():
                plt.imsave(str(output_subdir / f"{name.replace(' ', '_')}.png"),
                           cue_img, cmap='gray')

            overall_pbar.update(1)
            dataset_pbar.set_postfix(file=img_path.name)

        dataset_pbar.close()

    overall_pbar.close()
    print("Processing complete.")


# ── Augmentation factories ────────────────────────────────────────────────────

def imgaug_jpeg_distortion():
    """Random JPEG compression in quality range [20, 98]."""
    import imgaug.augmenters as iaa
    aug = iaa.JpegCompression(compression=(20, 98))
    def augment_fn(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(aug(image=rgb), cv2.COLOR_RGB2BGR)
    return augment_fn

def random_noising_augment():
    """Random Gaussian or Poisson colour noise."""
    import imgaug.augmenters as iaa
    poss = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=True)
    guss = iaa.AdditiveGaussianNoise(scale=0.2 * 255, per_channel=True)
    def augment_fn(img):
        return guss(image=img) if random.choice([True, False]) else poss(image=img)
    return augment_fn

def random_gray_noising_augment():
    """Convert to grayscale then add random Gaussian or Poisson noise."""
    import imgaug.augmenters as iaa
    to_gray = iaa.Grayscale(alpha=1.0)
    poss = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=False)
    guss = iaa.AdditiveGaussianNoise(scale=0.2 * 255, per_channel=False)
    def augment_fn(img):
        gray = to_gray(image=img)
        return guss(image=gray) if random.choice([True, False]) else poss(image=gray)
    return augment_fn


def _load_texture_paths(base_path):
    """Scan MLRN texture directory and return {category: [file_paths]}."""
    texture_map = {}
    for folder in ['A', 'S', 'C', 'F', 'M', 'W', 'O']:
        current = os.path.join(base_path, folder)
        if os.path.isdir(current):
            textures = [os.path.join(current, f) for f in os.listdir(current)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            if textures:
                texture_map[folder] = textures
    return texture_map

def create_old_photo_augmentor(mlrn_folder_path):
    """
    Factory: loads MLRN texture paths once and returns an augmentor that
    synthesises 'old photo' degradation via sepia/fade + scratch/stain textures.

    Args:
        mlrn_folder_path: root of the MLRN texture dataset (A, S, C, F, M, W, O subdirs)
    Returns:
        augment_fn(img_np: np.ndarray) -> np.ndarray
    """
    texture_map = _load_texture_paths(mlrn_folder_path)
    if not texture_map:
        return lambda img: img  # no-op if textures not found

    def augment_fn(img_np):
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), 'RGB')

        # Sepia tint or faded colour
        if random.random() < 0.5:
            img_aug = ImageOps.colorize(img.convert('L'),
                                        black=(50, 40, 30), white=(255, 240, 210))
        else:
            img_aug = ImageEnhance.Contrast(
                ImageEnhance.Color(img).enhance(random.uniform(0.1, 0.4))
            ).enhance(random.uniform(0.6, 0.9))

        # Structural damage: scratches (A), abrasions (S), creases (C)
        for _ in range(random.randint(1, 2)):
            cat = random.choice(['A', 'S', 'C'])
            if cat in texture_map:
                try:
                    tex = Image.open(random.choice(texture_map[cat])).convert('RGB')
                    img_aug = ImageChops.multiply(img_aug, tex.resize(img_aug.size, Image.Resampling.LANCZOS))
                except Exception:
                    pass

        # Surface stains: mildew (M), water (W), fade-off (F), other (O)
        for _ in range(random.randint(1, 2)):
            cat = random.choice(['F', 'M', 'W', 'O'])
            if cat in texture_map:
                try:
                    tex = Image.open(random.choice(texture_map[cat])).convert('RGB')
                    img_aug = Image.blend(img_aug, tex.resize(img_aug.size, Image.Resampling.LANCZOS),
                                          random.uniform(0.1, 0.35))
                except Exception:
                    pass

        return cv2.cvtColor(np.array(img_aug), cv2.COLOR_RGB2BGR)
    return augment_fn

def create_grayscale_augmentor():
    """Convert image to grayscale (three identical channels)."""
    def augment_fn(img_np):
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), 'RGB')
        return cv2.cvtColor(np.array(img.convert('L').convert('RGB')), cv2.COLOR_RGB2BGR)
    return augment_fn

def create_simple_degradation_augmentor():
    """
    Lightweight synthetic degradation pipeline:
    Gaussian blur → bicubic downsample → Gaussian noise → JPEG → bicubic upsample.
    Useful for generating paired LQ/HQ data without a full Real-ESRGAN setup.
    """
    try:
        from degradation_utils import img2tensor, tensor2img
    except ImportError:
        raise ImportError("degradation_utils.py must be in the same directory as this script.")

    SCALE_RANGE = [2, 4]
    JPEG_QUALITY_RANGE = [70, 95]
    NOISE_STD_RANGE = [1 / 255.0, 10 / 255.0]
    BLUR_SIGMA_RANGE = [0.2, 1.5]
    KERNEL_SIZE = 9

    def apply_jpeg(img_np, quality):
        ok, enc = cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img_np

    def gaussian_kernel(ksize, sigma):
        x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g2d = torch.outer(g, g)
        return (g2d / g2d.sum()).float()

    def augment_fn(gt_np):
        try:
            h, w = gt_np.shape[:2]
            scale = random.randint(*SCALE_RANGE)
            quality = random.randint(*JPEG_QUALITY_RANGE)
            noise_std = random.uniform(*NOISE_STD_RANGE)
            blur_sigma = random.uniform(*BLUR_SIGMA_RANGE)

            t = img2tensor(gt_np, bgr2rgb=True, float32=True)
            k = gaussian_kernel(KERNEL_SIZE, blur_sigma)
            pad = (KERNEL_SIZE - 1) // 2
            blurred = F.conv2d(
                F.pad(t.unsqueeze(0), (pad,)*4, mode='reflect'),
                k.view(1, 1, KERNEL_SIZE, KERNEL_SIZE).repeat(3, 1, 1, 1),
                padding=0, groups=3
            ).squeeze(0)
            lr = torch.clamp(
                F.interpolate(blurred.unsqueeze(0), scale_factor=1/scale,
                              mode='bicubic', align_corners=False).squeeze(0), 0, 1)
            noisy = torch.clamp(lr + torch.randn_like(lr) * noise_std, 0, 1)
            jpg = apply_jpeg(tensor2img(noisy, rgb2bgr=True, out_type=np.uint8), quality)
            return cv2.resize(jpg, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"Degradation augmentor error: {e}")
            return gt_np.astype(np.uint8)

    return augment_fn


# ── Paired-dataset path mappers ───────────────────────────────────────────────
# Each factory takes dataset-level arguments and returns a callable
# (path: str, index: int) -> target_path: str for use as `target_path_fn`.

def same_index_mapper_defocus(path):
    """blurry/ → sharp/ by sorted index (defocus dataset layout)."""
    mapping = sorted(Path(path.replace('blurry', 'sharp')).glob("*"))
    return lambda pth, ix: str(mapping[ix])

def same_index_mapper_ots(_path):
    """hazy/ → clear/ for the OTS haze dataset (stem-based filename matching)."""
    def map_fn(path, ix):
        p = Path(path)
        root = Path(str(path).split('hazy')[0]) / 'clear'
        stem = str(p.stem).split('_')[0] + "." + str(path).split('.')[-1]
        return str(root / stem)
    return map_fn

def same_index_mapper_sots(_path, old, new, outdoor=False):
    """SOTS hazy → gt by stripping the depth suffix from the filename."""
    def map_fn(path, ix):
        path = str(path).replace(old, new)
        return str(path).split('_')[0] + (Path(path).suffix if not outdoor else ".png")
    return map_fn

def same_index_mapper_sr_low():
    """RELLISUR LLLR low-res → NLHR/X4 high-res."""
    return lambda path, ix: str(Path((str(path).split('-')[0] + '.png').replace('LLLR', 'NLHR/X4')))

def same_index_mapper_replacer(path, old, new):
    """Simple single substring replacement in the full path."""
    return lambda pth, ix: str(Path(str(pth).replace(old, new)))

def same_index_mapper_replacerx(path, old, new):
    """Multiple simultaneous substring replacements in the full path."""
    def map_fn(pth, ix):
        pth = str(pth)
        for o, n in zip(old, new):
            pth = pth.replace(o, n)
        return str(Path(pth))
    return map_fn

def same_index_mapper_snow(path, old, new):
    """Replacement mapper that also converts .tif → .jpg extension."""
    return lambda pth, ix: str(Path(str(pth).replace(old, new))).replace('tif', 'jpg')

def same_index_mapper_sony_total(path, old, new):
    """Maps a Sony burst input directory to the single reference PNG in the target directory."""
    def map_fn(pth, ix):
        tpath = Path(str(Path(pth).parent).replace(old, new))
        return str(list(tpath.glob('*.png'))[0])
    return map_fn

def same_index_mapper_shadow(path, old, new, add):
    """Replacement mapper with an extra suffix appended to the stem."""
    def map_fn(pth, ix):
        tpath = Path(str(pth).replace(old, new))
        return str(tpath.parent / (str(tpath.stem) + add))
    return map_fn

def same_index_mapper_overexposure(path, old, new):
    """Maps over-exposed frames to their reference by stripping the exposure index suffix."""
    def map_fn(pth, ix):
        tpath = Path(str(pth).replace(old, new))
        return str(tpath.parent / ("_".join(str(tpath.stem).split('_')[:-1]) + ".jpg"))
    return map_fn


# ── Dataset configuration ─────────────────────────────────────────────────────
# Populate this list before running. Each entry is a list containing one config dict.
# See process_datasets() docstring for the full set of supported keys.

dataset_configs = [
    [
        {
            'root': '/path/to/degradations',
            'dataset_root': '/path/to/degradations/your_dataset',
            'augment_fn': lambda x: x,
            'name': 'your_dataset',
            'target_path_fn': same_index_mapper_replacer(
                '/path/to/degradations/your_dataset', 'input_subdir', 'target_subdir'),
        }
    ],
]

if __name__ == '__main__':
    process_datasets(
        dataset_configs=dataset_configs,
        output_dir='/path/to/output',
        target_size=512,
        generate_features=False,
        generate_captions=False,
    )
