import pandas as pd
from joblib import load

pipe = load('../model/laptop_price_model.joblib')

data = {
    'Brand':'HP',
    'Processor':'Core i3',
    'Operating_System':'Windows 11 Home',
    'Storage':512,
    'RAM':8,
    'Touch_Screen':0
}

df_input = pd.DataFrame([data])

result = pipe.predict(df_input)[0]
print(f"Predicted Price: ₹{round(result,2)}")
