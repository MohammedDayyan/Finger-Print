from fastapi import FastAPI, UploadFile , File 
from fastapi.middleware.cors import CORSMiddleware
import os 
import shutil
from ml.inference import predict_fingerprint

backend = FastAPI(title="Fingerprint Prediction API")
backend.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)


@backend.get('/')
def home():
    return {"message": "Welcome to fingerprint detection API."}


@backend.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    file_path = os.path.join(UPLOAD_FOLDER,file.filename)
    
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    result = predict_fingerprint(file_path)
    
    os.remove(file_path)
    
    return {"prediction": result}

