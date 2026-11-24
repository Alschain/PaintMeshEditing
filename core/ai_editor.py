"""Pluggable 2D AI image editing backends and mask utilities for PaintMeshEditing.

Backends supported:
  - local_refine              : Zero-dependency OpenCV color/morphology label refinement & auto-masking
  - a1111_api                 : Automatic1111 Stable Diffusion WebUI REST API (/sdapi/v1/img2img)
  - diffusers_inpaint         : Local HuggingFace Diffusers StableDiffusionInpaintPipeline
  - diffusers_instruct_pix2pix: Local HuggingFace Diffusers StableDiffusionInstructPix2PixPipeline
  - custom_http_api           : Generic JSON REST endpoint ({image_b64, mask_b64, prompt, ...})
"""

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np


def _encode_png_b64(img_bgr_or_gray):
    ok, buf = cv2.imencode('.png', img_bgr_or_gray)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG.")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _decode_png_b64(b64_str):
    if ',' in b64_str and b64_str.strip().startswith('data:'):
        b64_str = b64_str.split(',', 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image returned from AI backend.")
    return img


def compute_edit_mask(
    rendered_bgr,
    edited_bgr,
    painted_mask=None,
    label_color_bgr=None,
    color_tol=40,
    diff_thresh=15,
    dilate_px=5,
    fg_mask=None,
):
    """Automatically compute the PME optimization mask (uint8, 0 or 255).

    Following the PME workflow, the mask is the union of:
      1. Pixels that changed between the rendered image and the edited target image
      2. Optional user-painted mask strokes
      3. Optional label-colored pixels in BOTH the original render (within foreground) and the edited image
    """
    h, w = edited_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if rendered_bgr is not None and rendered_bgr.shape[:2] == (h, w):
        diff = np.abs(edited_bgr.astype(np.int16) - rendered_bgr.astype(np.int16)).max(axis=-1)
        mask[diff > int(diff_thresh)] = 255

    if painted_mask is not None:
        pm = painted_mask
        if pm.ndim == 3:
            pm = pm[..., 0]
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[pm > 127] = 255

    if label_color_bgr is not None:
        target = np.array(label_color_bgr, dtype=np.int16).reshape(1, 1, 3)
        tol = int(color_tol)
        fg_bool = None
        if fg_mask is not None:
            fm = fg_mask[..., 0] if fg_mask.ndim == 3 else fg_mask
            if fm.shape[:2] != (h, w):
                fm = cv2.resize(fm, (w, h), interpolation=cv2.INTER_NEAREST)
            fg_bool = fm > 127

        if rendered_bgr is not None and rendered_bgr.shape[:2] == (h, w):
            dist_r = np.abs(rendered_bgr.astype(np.int16) - target).max(axis=-1)
            match_r = dist_r <= tol
            if fg_bool is not None:
                match_r = match_r & fg_bool
            elif int(target.max()) <= tol:
                # Avoid matching pure-black background when no explicit fg_mask is provided
                match_r = match_r & (rendered_bgr.max(axis=-1) > 0)
            mask[match_r] = 255

        dist_e = np.abs(edited_bgr.astype(np.int16) - target).max(axis=-1)
        match_e = dist_e <= tol
        if int(target.max()) <= tol:
            # For dark label colors, only match edited pixels that differ from background or lie in foreground
            if fg_bool is not None:
                match_e = match_e & fg_bool
            else:
                match_e = match_e & (edited_bgr.max(axis=-1) > 0)
        mask[match_e] = 255

    # Clean noise and dilate boundary slightly so differentiable renderer has smooth support
    if dilate_px and int(dilate_px) > 0:
        k = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


class AIImageEditor:
    """Unified interface for 2D image editing models in the PME pipeline."""

    BACKENDS = (
        'local_refine',
        'a1111_api',
        'diffusers_inpaint',
        'diffusers_instruct_pix2pix',
        'custom_http_api',
    )

    def __init__(self):
        self._cached_pipelines = {}

    def edit(
        self,
        image_bgr,
        mask_gray=None,
        backend='local_refine',
        prompt='',
        negative_prompt='',
        strength=0.65,
        guidance_scale=7.5,
        steps=25,
        endpoint='http://127.0.0.1:7860',
        api_key='',
        model_id=None,
        device='cuda',
        seed=-1,
        extra_params=None,
    ):
        """Run 2D image editing and return (edited_bgr, updated_mask_gray)."""
        backend = (backend or 'local_refine').lower()
        extra_params = extra_params or {}

        if image_bgr is None:
            raise ValueError("Input image is None.")

        h, w = image_bgr.shape[:2]
        if mask_gray is None:
            mask_u8 = np.full((h, w), 255, dtype=np.uint8)
        else:
            mask_u8 = mask_gray[..., 0] if mask_gray.ndim == 3 else mask_gray.copy()
            if mask_u8.shape[:2] != (h, w):
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_u8 = np.where(mask_u8 > 127, 255, 0).astype(np.uint8)

        if backend == 'local_refine':
            edited = self._run_local_refine(image_bgr, mask_u8, prompt=prompt, strength=strength)
        elif backend == 'a1111_api':
            edited = self._run_a1111_api(
                image_bgr,
                mask_u8,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                steps=steps,
                endpoint=endpoint,
                seed=seed,
            )
        elif backend == 'diffusers_inpaint':
            edited = self._run_diffusers_inpaint(
                image_bgr,
                mask_u8,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                steps=steps,
                model_id=model_id or 'runwayml/stable-diffusion-inpainting',
                device=device,
                seed=seed,
            )
        elif backend == 'diffusers_instruct_pix2pix':
            edited = self._run_diffusers_instruct_pix2pix(
                image_bgr,
                mask_u8,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                steps=steps,
                model_id=model_id or 'timbrooks/instruct-pix2pix',
                device=device,
                seed=seed,
            )
        elif backend == 'custom_http_api':
            edited = self._run_custom_http_api(
                image_bgr,
                mask_u8,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                steps=steps,
                endpoint=endpoint,
                api_key=api_key,
                extra_params=extra_params,
            )
        else:
            raise ValueError(f"Unsupported AI editor backend: {backend}")

        if edited.shape[:2] != (h, w):
            edited = cv2.resize(edited, (w, h), interpolation=cv2.INTER_LINEAR)

        # Preserve unmasked pixels when a partial mask is provided
        if mask_gray is not None and np.any(mask_u8 == 0) and np.any(mask_u8 > 0):
            alpha = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (7, 7), 0)[..., None]
            edited = np.clip(edited.astype(np.float32) * alpha + image_bgr.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

        updated_mask = compute_edit_mask(image_bgr, edited, painted_mask=mask_u8, dilate_px=3)
        return edited, updated_mask

    def _run_local_refine(self, image_bgr, mask_u8, prompt='', strength=0.65):
        """Offline edge-preserving / morphological refinement for quick local prototyping."""
        cmd = (prompt or '').strip().lower()
        k = max(3, int(round(strength * 15)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        if 'erode' in cmd or 'thin' in cmd or 'shrink' in cmd or '瘦' in cmd or '细' in cmd:
            processed = cv2.erode(image_bgr, kernel, iterations=1)
        elif 'dilate' in cmd or 'thick' in cmd or 'expand' in cmd or '胖' in cmd or '粗' in cmd:
            processed = cv2.dilate(image_bgr, kernel, iterations=1)
        elif 'sharpen' in cmd or 'detail' in cmd or '锐化' in cmd:
            blur = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=max(1.0, strength * 3.0))
            processed = cv2.addWeighted(image_bgr, 1.0 + strength, blur, -strength, 0)
        elif 'inpaint' in cmd or 'remove' in cmd or '擦除' in cmd or '修复' in cmd:
            processed = cv2.inpaint(image_bgr, mask_u8, inpaintRadius=max(3, k // 2), flags=cv2.INPAINT_TELEA)
        else:
            # Default: bilateral smooth inside masked region to clean hand-painted strokes
            d = max(5, k)
            sigma = 25.0 + strength * 75.0
            processed = cv2.bilateralFilter(image_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)

        return processed

    def _run_a1111_api(
        self,
        image_bgr,
        mask_u8,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        steps,
        endpoint,
        seed,
    ):
        url = endpoint.rstrip('/') + '/sdapi/v1/img2img'
        h, w = image_bgr.shape[:2]
        # Keep dimensions multiples of 8 for Stable Diffusion
        sd_w = max(64, (w // 8) * 8)
        sd_h = max(64, (h // 8) * 8)
        img_resized = cv2.resize(image_bgr, (sd_w, sd_h), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_u8, (sd_w, sd_h), interpolation=cv2.INTER_NEAREST)

        payload = {
            'init_images': [_encode_png_b64(img_resized)],
            'mask': _encode_png_b64(mask_resized),
            'prompt': prompt or '',
            'negative_prompt': negative_prompt or '',
            'denoising_strength': float(strength),
            'cfg_scale': float(guidance_scale),
            'steps': int(steps),
            'width': sd_w,
            'height': sd_h,
            'seed': int(seed),
            'inpainting_fill': 1,  # original
            'inpaint_full_res': False,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not connect to Automatic1111 WebUI API at {url}. "
                f"Make sure WebUI is launched with `--api`. Details: {exc}"
            ) from exc

        images = data.get('images')
        if not images:
            raise RuntimeError(f"A1111 API returned no images: {data}")
        return _decode_png_b64(images[0])

    def _run_custom_http_api(
        self,
        image_bgr,
        mask_u8,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        steps,
        endpoint,
        api_key,
        extra_params,
    ):
        if not endpoint:
            raise ValueError("Custom HTTP API requires a valid endpoint URL.")
        payload = {
            'image_b64': _encode_png_b64(image_bgr),
            'mask_b64': _encode_png_b64(mask_u8),
            'prompt': prompt or '',
            'negative_prompt': negative_prompt or '',
            'strength': float(strength),
            'guidance_scale': float(guidance_scale),
            'steps': int(steps),
        }
        payload.update(extra_params or {})

        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Custom AI HTTP endpoint request failed ({endpoint}): {exc}") from exc

        b64 = data.get('image_b64') or (data.get('images', [None])[0] if isinstance(data.get('images'), list) else None)
        if not b64:
            raise RuntimeError(f"Custom API response must contain 'image_b64' or 'images': {list(data.keys())}")
        return _decode_png_b64(b64)

    def _run_diffusers_inpaint(
        self,
        image_bgr,
        mask_u8,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        steps,
        model_id,
        device,
        seed,
    ):
        try:
            import torch
            from PIL import Image
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:
            raise ImportError(
                "Local diffusers backend requires `diffusers`, `transformers`, `accelerate`, `torch`, and `Pillow`.\n"
                "Install with: pip install diffusers transformers accelerate pillow"
            ) from exc

        key = ('inpaint', model_id, device)
        pipe = self._cached_pipelines.get(key)
        if pipe is None:
            dtype = torch.float16 if ('cuda' in device and torch.cuda.is_available()) else torch.float32
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe = pipe.to(device if torch.cuda.is_available() else 'cpu')
            self._cached_pipelines[key] = pipe

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((512, 512), Image.BICUBIC)
        pil_mask = Image.fromarray(mask_u8).resize((512, 512), Image.NEAREST)

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

        out = pipe(
            prompt=prompt or '',
            negative_prompt=negative_prompt or None,
            image=pil_img,
            mask_image=pil_mask,
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]

        out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        return out_bgr

    def _run_diffusers_instruct_pix2pix(
        self,
        image_bgr,
        mask_u8,
        prompt,
        negative_prompt,
        guidance_scale,
        steps,
        model_id,
        device,
        seed,
    ):
        try:
            import torch
            from PIL import Image
            from diffusers import StableDiffusionInstructPix2PixPipeline
        except ImportError as exc:
            raise ImportError(
                "Local InstructPix2Pix backend requires `diffusers`, `transformers`, `accelerate`, `torch`, and `Pillow`.\n"
                "Install with: pip install diffusers transformers accelerate pillow"
            ) from exc

        key = ('ip2p', model_id, device)
        pipe = self._cached_pipelines.get(key)
        if pipe is None:
            dtype = torch.float16 if ('cuda' in device and torch.cuda.is_available()) else torch.float32
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe = pipe.to(device if torch.cuda.is_available() else 'cpu')
            self._cached_pipelines[key] = pipe

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((512, 512), Image.BICUBIC)

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

        out = pipe(
            prompt=prompt or '',
            negative_prompt=negative_prompt or None,
            image=pil_img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]

        out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        return out_bgr
