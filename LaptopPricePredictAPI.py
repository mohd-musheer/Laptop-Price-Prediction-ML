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
def home():
    with open("index.html", "r") as f:
        return f.read()

# Prediction Route
@app.post("/predict")
def predict_price(data: DataClass):
    try:
        model = joblib.load("LaptopPricePredict.pkl")

        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        # 🔥 FIX HERE: Convert numpy.int64 → int
        prediction = int(prediction)

        return {"prediction": prediction}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
