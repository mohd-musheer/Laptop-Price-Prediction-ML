FROM python:3.10-slim

WORKDIR /LaptopPricePredict

COPY . .

RUN pip install --no-cache-dir -r requirments.txt

ENV PORT=10000

EXPOSE 8086

CMD ["uvicorn","LaptopPricePredictAPI:app","--host","0.0.0.0","--port","8086"]