import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from joblib import dump

df = pd.read_csv('../data/laptops.csv')

x = df[['Brand','Processor','Operating_System','Storage','RAM','Touch_Screen']]
y = df['Price']

preprocess = ColumnTransformer([
    ('num',StandardScaler(),['Storage','RAM']),
    ('cat',OneHotEncoder(handle_unknown='ignore'),['Brand','Processor','Operating_System','Touch_Screen'])
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('pca', PCA(n_components=2)),
    ('model', RandomForestRegressor(random_state=42))
])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pipe.fit(x_train,y_train)
pred = pipe.predict(x_test)

print("R² Score:", round(r2_score(y_test,pred),2)*100,"%")

dump(pipe, '../model/laptop_price_model.joblib')
print("Model saved in /model folder")
