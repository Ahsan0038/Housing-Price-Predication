import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Fetch dataset
california = fetch_california_housing(as_frame=True)

df = california.data.copy()
df['Target'] = california.target

# Selected features
selected_features = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'Population',
    'Latitude',
    'Longitude'
]

X = df[selected_features]
y = df['Target']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=159,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

print(f"Model Accuracy: {accuracy:.2f}%")

# Save model
with open("california_housing_model.pkl", "wb") as f:
    pickle.dump(model, f)