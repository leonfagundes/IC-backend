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
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "modelo_mobilenetv2_128x128_finetuned.keras")

def load_model():
    global MODEL
    MODEL = tf.keras.models.load_model(MODEL_PATH)

def is_grayscale_image(image: Image.Image) -> bool:
    """
    Verifica se a imagem é em escala de cinza.
    Retorna True se todos os pixels têm R=G=B (tons de cinza).
    """
    # Converte para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Converte para array numpy
    img_array = np.array(image)
    
    # Verifica se R=G=B para todos os pixels
    # Em uma imagem grayscale verdadeira, os canais R, G e B são idênticos
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    # Verifica se todos os pixels têm R=G=B
    is_gray = np.all(r_channel == g_channel) and np.all(g_channel == b_channel)
    
    return is_gray

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
    allow_methods=["POST"],
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
        
        if not is_grayscale_image(image):
            raise HTTPException(status_code=400, detail="Arquivo deve ser ressonancia")
        
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("ERRO DETALHADO:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")
