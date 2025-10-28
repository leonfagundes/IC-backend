from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

MODEL = None
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
TARGET_SIZE = (128, 128)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "modelo_mobilenetv2_96x96_finetuned.keras")

def load_model():
    global MODEL
    MODEL = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(TARGET_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        predictions = MODEL.predict(processed_image, verbose=0)
        pred_vector = predictions[0]
        predicted_idx = int(np.argmax(pred_vector))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(pred_vector[predicted_idx])
        
        probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, pred_vector)
        }
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")
