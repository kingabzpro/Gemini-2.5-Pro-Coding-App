import io
import os
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from torchvision import models

# Load environment variables from .env file
if not load_dotenv():
    raise ValueError("Failed to load .env file")

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set in .env file")

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR10 Image Classification APP",
    description="A production-ready API for image classification using a fine-tuned model on CIFAR10.",
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Define CIFAR10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned ResNet18 model
model_path = "finetuned_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# API key validation dependency function
def get_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        logger.warning("API key is missing")
        raise HTTPException(status_code=403, detail="API key is missing")
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key}")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    """Endpoint for checking if the API is running."""
    return {"status": "ok", "message": "API is running", "device": str(device)}


@app.get("/model-info", summary="Get Model Information", tags=["Metadata"])
async def get_model_info(api_key: str = Depends(get_api_key)):
    """Combined endpoint for retrieving model metadata and class names."""
    model_info = {
        "model_architecture": "ResNet18",
        "num_classes": num_classes,
        "class_names": class_names,
        "device": str(device),
        "model_weights_file": model_path,
        "description": "Model fine-tuned on CIFAR10 dataset",
    }
    return JSONResponse(model_info)


@app.post("/predict", summary="Predict Image Class", tags=["Inference"])
async def predict(
    file: UploadFile = File(...),
    include_confidence: bool = Query(
        False, description="Include confidence scores for top predictions"
    ),
    top_k: int = Query(
        1, ge=1, le=10, description="Number of top predictions to return"
    ),
    api_key: str = Depends(get_api_key),
):
    """
    Unified prediction endpoint that can return either simple class prediction
    or detailed predictions with confidence scores.
    """
    # Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        logger.error(f"Invalid file format: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only PNG and JPEG are supported.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        logger.error("Uploaded file is not a valid image")
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image.")

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = model(input_tensor)

            if include_confidence:
                # Return predictions with confidence scores
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_idxs = torch.topk(
                    probabilities, k=min(top_k, num_classes)
                )
                top_probs = top_probs.cpu().numpy().tolist()[0]
                top_idxs = top_idxs.cpu().numpy().tolist()[0]
                predictions = [
                    {"class": class_names[idx], "confidence": prob}
                    for idx, prob in zip(top_idxs, top_probs)
                ]
                return JSONResponse({"predictions": predictions})
            else:
                # Return simple prediction (just the class)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds[0]]
                return JSONResponse({"predicted_class": predicted_class})
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during model inference.")


if __name__ == "__main__":
    uvicorn.run("secure_app:app",port=5454)
