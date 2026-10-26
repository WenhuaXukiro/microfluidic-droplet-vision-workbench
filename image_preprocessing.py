"""Versioned, geometry-preserving enhancement shared by preview and classifiers."""

PREPROCESS_VERSION = 1
PRESETS = {
    "none": {"label": "保持原样", "description": "保留当前处理图像素"},
    "clahe_soft": {"label": "温和局部对比度", "description": "CLAHE限幅1.5、8×8网格，与原亮度混合35%"},
    "illumination_soft": {"label": "温和光照校正", "description": "低频背景校正，限制增益并与原亮度混合50%"},
    "denoise_detail": {"label": "保边去噪与轻锐化", "description": "轻度双边去噪与受限细节增强，保留颗粒边缘"},
}


def validate_preset(preset):
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("识别增强方案无效")
    return preset


def enhance_image(image, preset="none"):
    """Enhance luminance only. Never crop, resize, segment or change labels."""
    from PIL import Image, ImageOps
    preset = validate_preset(preset)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if preset == "none":
        return image
    import cv2
    import numpy as np
    rgb = np.asarray(image)
    channels = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    original = channels[:, :, 0].astype(np.float32)
    if preset == "clahe_soft":
        local = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(channels[:, :, 0])
        enhanced = .65 * original + .35 * local
    elif preset == "illumination_soft":
        sigma = max(8., min(image.size) / 12.)
        background = cv2.GaussianBlur(original, (0, 0), sigmaX=sigma, sigmaY=sigma)
        reference = float(np.median(background))
        gain = np.clip(reference / np.maximum(background, 16.), .8, 1.25)
        enhanced = original * (.5 + .5 * gain)
    else:
        filtered = cv2.bilateralFilter(channels[:, :, 0], 5, 8., 2.).astype(np.float32)
        base = .7 * original + .3 * filtered
        detail = base - cv2.GaussianBlur(base, (0, 0), sigmaX=.8, sigmaY=.8)
        detail = np.where(np.abs(detail) >= 1.5, detail, 0.)
        enhanced = original + np.clip(base + .35 * detail - original, -12., 12.)
    channels[:, :, 0] = np.clip(np.rint(enhanced), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB))


def preview_payload(payload):
    import base64
    import io
    from PIL import Image
    preset = validate_preset(payload.get("preprocessing", "none"))
    raw = base64.b64decode(payload["data"], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("增强预览最多1600万像素，请缩小裁剪区域或输出倍率")
        enhanced = enhance_image(image, preset)
    buffer = io.BytesIO()
    enhanced.save(buffer, format="PNG")
    return {"data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "width": enhanced.width, "height": enhanced.height,
            "preprocessing": preset, "label": PRESETS[preset]["label"],
            "preprocessing_version": PREPROCESS_VERSION}
