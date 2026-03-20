from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import DamageNet as DamageCNN

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DamageCNN(num_classes=3)
model.load_state_dict(torch.load("imporved_damage_model.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
classes = ["minor", "moderate", "severe"]

# SAME preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---- Test Time Augmentation ----
def tta_predict(image_tensor):
    """
    Perform inference with test-time augmentation.
    """
    with torch.no_grad():

        # Original
        pred1 = F.softmax(model(image_tensor), dim=1)

        # Horizontal flip
        flipped = torch.flip(image_tensor, dims=[3])
        pred2 = F.softmax(model(flipped), dim=1)

        # Average predictions
        pred = (pred1 + pred2) / 2

    return pred


@app.get("/")
def root():
    return {"message": "Car Damage Severity Detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Load image
    image = Image.open(file.file).convert("RGB")

    # Preprocess
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    pred = tta_predict(image)

    # Get class + confidence
    confidence, predicted_class = torch.max(pred, 1)

    return {
        "prediction": classes[predicted_class.item()],
        "confidence": float(confidence.item()),
        "probabilities": {
            classes[i]: float(pred[0][i]) for i in range(len(classes))
        }
    }