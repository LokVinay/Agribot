from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware  # ✅ STEP 1

app = FastAPI()

# ✅ STEP 2: Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change "*" to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
with open("crop_recommendation_model.pkl", "rb") as file:
    model = pickle.load(file)

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/recommend")
def recommend_crop(data: CropInput):
    try:
        input_data = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
        prediction = model.predict(input_data)[0]
        return {"recommended_crop": prediction}
    except Exception as e:
        return {"error": str(e)}
