
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


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "clinguard_resnet18_best.pth"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_FILE_SIZE = 10 * 1024 * 1024
IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ClinGuard MRI API",
    description=(
        "Research prototype for brain MRI classification "
        "and Grad-CAM visualization."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://brain-recuhb5vs-mitochondria3.vercel.app",
        "https://brain-mri-sigma.vercel.app",
        "https://brain-mri-git-main-mitochondria3.vercel.app",
    ],
    allow_origin_regex=r"https://brain-[a-zA-Z0-9-]+-mitochondria3\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)



# --------------------------------------------------
# Load trained ResNet18
# --------------------------------------------------

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(CLASSES),
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Final convolutional block used for Grad-CAM.
target_layer = model.layer4[-1]

# Prevent simultaneous requests from interfering with
# the shared model or Grad-CAM hooks.
inference_lock = threading.Lock()


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

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


# --------------------------------------------------
# Grad-CAM visualization
# --------------------------------------------------

def create_gradcam(input_tensor, original_image, class_index):
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(
        save_activation
    )

    try:
        # Grad-CAM needs gradients, so this must not run
        # inside torch.no_grad() or inference_mode().
        with torch.enable_grad():
            logits = model(input_tensor)
            activation = captured["activation"]

            # Explain the selected class's raw score.
            score = logits[0, class_index]

            gradient = torch.autograd.grad(
                outputs=score,
                inputs=activation,
                retain_graph=False,
                create_graph=False,
            )[0]

        activation = activation.detach()
        gradient = gradient.detach()

        # Calculate channel importance weights.
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of feature maps.
        cam = (
            weights * activation
        ).sum(dim=1)[0].relu()

        # Upsample to the input image size.
        cam = F.interpolate(
            cam[None, None],
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Normalize for display.
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        cam = cam.cpu().numpy()

        # Suppress weaker activations for a cleaner display.
        # This is a visualization choice, not a validated
        # medical threshold.
        threshold = 0.30

        opacity = np.clip(
            (cam - threshold) / (1.0 - threshold),
            0.0,
            1.0,
        ) * 0.35

        # Resize the original MRI for the overlay.
        source = np.asarray(
            original_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            dtype=np.float32,
        )

        # Red heatmap overlay.
        red = np.zeros_like(source)
        red[:, :, 0] = 255

        alpha = opacity[:, :, None]

        overlay = (
            source * (1.0 - alpha)
            + red * alpha
        )

        overlay = np.clip(
            overlay, 0, 255
        ).astype(np.uint8)

        # Encode as a JPEG data URL for the frontend.
        buffer = BytesIO()

        Image.fromarray(overlay).save(
            buffer,
            format="JPEG",
            quality=90,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        return "data:image/jpeg;base64," + encoded

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


# --------------------------------------------------
# MRI prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate declared upload type.
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    # Read at most 10 MB plus one byte.
    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty upload.",
        )

    # Decode and preprocess the image.
    try:
        with Image.open(BytesIO(contents)) as image:
            original_image = image.convert("RGB")

            tensor = preprocess(
                original_image
            ).unsqueeze(0).to(device)

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Classification and Grad-CAM use the same model.
    try:
        with inference_lock:

            # Standard classification inference.
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(
                    logits, dim=1
                )[0].cpu()

            predicted_index = int(
                probabilities.argmax()
            )

            # Explain the predicted class.
            gradcam = create_gradcam(
                tensor,
                original_image,
                predicted_index,
            )

    except Exception:
        # Keep internal tracebacks out of API responses.
        # Inspect server logs if inference fails.
        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed. "
                "Check the backend runtime logs."
            ),
        )

    return {
        "prediction": CLASSES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]), 4
        ),
        "probabilities": {
            name: round(
                float(probabilities[i]), 4
            )
            for i, name in enumerate(CLASSES)
        },
        "model": "ResNet18",
        "gradcam": gradcam,
        "explanation_notice": (
            "Grad-CAM is an experimental, coarse visualization "
            "of regions influencing the model score. It is not "
            "tumor segmentation or proof of clinically meaningful "
            "reasoning. Red regions do not establish tumor location."
        ),
        "notice": (
            "Research output only. Not a medical diagnosis. "
            "Have MRI findings assessed by a qualified clinician."
        ),
    }
