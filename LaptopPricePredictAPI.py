from fastapi import FastAPI
from pydantic import BaseModel ,Field
from fastapi.responses import JSONResponse,HTMLResponse
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles

app=FastAPI()

class DataClass(BaseModel):
    Brand:str=Field(...,example='HP',description='HP Apple Lenovo ASUS DELL Acer SAMSUNG MSI Infinix Ultimus CHUWI WINGS ZEBRONICS Primebook GIGABYTE realme MICROSOFT LG')
    Processor:str=Field(...,example='Core i3',description='Core i3, M1, Core i7, Core i5, Ryzen 5 Hexa Core,Celeron Dual Core, Ryzen 7 Octa Core, Ryzen 5 Quad Core,Ryzen 3 Dual Core, Ryzen 3 Quad Core, M2, Celeron Quad Core,Athlon Dual Core, MediaTek Kompanio 1200, Ryzen 9 Octa Core,MediaTek MT8788, Ryzen Z1 HexaCore, MediaTek Kompanio 500, Core i9,MediaTek Kompanio 520, Ryzen Z1 Octa Core, Pentium Silver, Ryzen 5,M1 Max, M2 Max, M3 Pro, M1 Pro, Ryzen 7 Quad Core,Ryzen 5 Dual Core, Ryzen 9 16 Core')
    Operating_System:str=Field(...,example='Windows 11 Home',description='Windows 11 Home, Mac OS Big Sur, DOS, Mac OS Monterey, Chrome,Windows 10, Windows 10 Home, Prime OS, Windows 11 Pro, Ubuntu,Windows 10 Pro, macOS Ventura, macOS Sonoma, Mac OS Mojave')
    Storage:int=Field(...,example=512,description=' in GB')
    RAM:int=Field(...,example=8,description=' in GB')
    Touch_Screen:int=Field(...,example=0,description=' in 0 1 for yes or no')



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r") as file:
        return file.read()


@app.post('/predict')
def pred_price(d:DataClass):
    pipe=joblib.load('LaptopPricePredict.pkl')

    data = {
    'Brand': d.Brand,
    'Processor': d.Processor,
    'Operating_System':d.Operating_System,
    'Storage': d.Storage,
    'RAM': d.RAM,
    'Touch_Screen': d.Touch_Screen
    }
    df = pd.DataFrame([data])
    result= pipe.predict(df)[0]
    return JSONResponse(status_code=200,content={'message':f'Predicted Price is : {result}'})

