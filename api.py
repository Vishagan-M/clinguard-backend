
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "clinguard_resnet18_best.pth"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(
    title="ClinGuard MRI API",
    description="Research prototype for brain MRI image classification.",
    version="1.0.0",
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


@app.get("/")
def home():
    return {
        "service": "ClinGuard MRI API",
        "status": "ready",
        "device": str(device),
        "classes": CLASSES,
        "notice": "Research prototype; not for medical diagnosis.",
    }


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
            image = image.convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax())

    return {
        "prediction": CLASSES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 4),
        "probabilities": {
            name: round(float(probabilities[i]), 4)
            for i, name in enumerate(CLASSES)
        },
        "model": "ResNet18",
        "notice": (
            "Research output only. Not a medical diagnosis. "
            "Have MRI findings assessed by a qualified clinician."
        ),
    }
