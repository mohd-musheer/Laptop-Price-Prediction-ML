from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Allow frontend → backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataClass(BaseModel):
    Brand: str
    Processor: str
    Operating_System: str
    Storage: int
    RAM: int
    Touch_Screen: int

# Home Route
@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>Laptop Price Predictor API is Live 🚀</h2>"

# Prediction Route
@app.post("/predict")
def predict_price(data: DataClass):
    model = joblib.load("LaptopPricePredict.pkl")

    df = pd.DataFrame([{
        "Brand": data.Brand,
        "Processor": data.Processor,
        "Operating_System": data.Operating_System,
        "Storage": data.Storage,
        "RAM": data.RAM,
        "Touch_Screen": data.Touch_Screen
    }])

    try:
        prediction = model.predict(df)[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
