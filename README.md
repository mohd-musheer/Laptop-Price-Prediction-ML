📌 Laptop Price Prediction (ML Model)

This project predicts the price of a laptop based on its specifications using Machine Learning.
The model is trained using Random Forest Regressor + PCA, achieving ~80% R² Accuracy.

Running with UI : https://laptop-price-prediction-vhid.onrender.com



api : https://laptop-price-prediction-vhid.onrender.com/predict

🧠 Summary

Input features: Brand, Processor, Operating System, Storage, RAM, Touch Screen

Preprocessing: StandardScaler + OneHotEncoder

Model used: RandomForestRegressor

Dimensionality Reduction: PCA (2 components)

Final Accuracy (R² Score): ~80%

🚀 How to Run
Install Requirements
pip install -r requirements.txt

Train Model
python src/train.py

Predict Price

Edit data in src/predict.py, then:

python src/predict.py

📊 R² Score
Model Accuracy (R²): ~80%

📂 Important Files
File	Use
train.py	Train and save model
predict.py	Predict new laptop price
laptop_price_model.joblib	Saved trained model
✨ Example Input
{'Brand':'HP','Processor':'Core i3','Operating_System':'Windows 11 Home','Storage':512,'RAM':8,'Touch_Screen':0}

✍️ Author

Musheer — ML Student building real-world projects 🚀