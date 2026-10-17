
from io import BytesIO
from pathlib import Path
import base64
import threading

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "clinguard_resnet18_best.pth"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_FILE_SIZE = 10 * 1024 * 1024
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(
    title="ClinGuard MRI API",
    description="Research prototype for brain MRI classification and Grad-CAM.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://brain-mri-sigma.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ResNet18's final convolutional block provides the Grad-CAM activations.
target_layer = model.layer4[-1]
inference_lock = threading.Lock()


@app.get("/")
def home():
    return {
        "service": "ClinGuard MRI API",
        "status": "ready",
        "device": str(device),
        "classes": CLASSES,
        "features": ["classification", "gradcam"],
        "notice": "Research prototype; not for medical diagnosis.",
    }


def create_gradcam(input_tensor, original_image, class_index):
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(save_activation)

    try:
        # Grad-CAM requires gradients; do not use inference_mode here.
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, class_index]
            score.backward()

        activation = captured["activation"].detach()
        gradient = captured["activation"].grad.detach()

        # Average gradients spatially to get one weight per channel.
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True).relu()

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)

        heat = (cam * 255).byte().cpu().numpy()
        heat_image = Image.fromarray(heat, mode="L").convert("RGB")

        # Simple red heatmap blended over the resized source MRI.
        heat_array = np.asarray(heat_image, dtype=np.float32)
        red_overlay = np.zeros_like(heat_array)
        red_overlay[:, :, 0] = heat_array[:, :, 0]
        red_overlay[:, :, 1] = heat_array[:, :, 0] * 0.12
        red_overlay[:, :, 2] = heat_array[:, :, 0] * 0.08
        red_image = Image.fromarray(
            np.clip(red_overlay, 0, 255).astype(np.uint8)
        )

        source = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        overlay = Image.blend(source, red_image, alpha=0.38)

        buffer = BytesIO()
        overlay.save(buffer, format="JPEG", quality=85)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + encoded

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        with Image.open(BytesIO(contents)) as image:
            original_image = image.convert("RGB")
            tensor = preprocess(original_image).unsqueeze(0).to(device)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        # Protect the shared model and its temporary gradient hooks.
        with inference_lock:
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu()

            predicted_index = int(probabilities.argmax())

            gradcam = create_gradcam(
                tensor,
                original_image,
                predicted_index,
            )

    except Exception:
        # Keep internal tracebacks out of the public API response.
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Check the backend runtime logs.",
        )

    return {
        "prediction": CLASSES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 4),
        "probabilities": {
            name: round(float(probabilities[i]), 4)
            for i, name in enumerate(CLASSES)
        },
        "model": "ResNet18",
        "gradcam": gradcam,
        "explanation_notice": (
            "Grad-CAM is a coarse visualization of regions influencing "
            "the model score. It is not a tumor segmentation or proof "
            "of clinically meaningful reasoning."
        ),
        "notice": (
            "Research output only. Not a medical diagnosis. "
            "Have MRI findings assessed by a qualified clinician."
        ),
    }
