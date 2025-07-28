

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import matplotlib.pyplot as plt
import warnings
import streamlit as st
st.title("Farmer Income Predictor")
warnings.filterwarnings("ignore")

# 1. Synthetic Data Generation
np.random.seed(42)
N = 500
states = ['Punjab', 'Maharashtra', 'UP', 'Bihar', 'Tamil Nadu', 'MP', 'Rajasthan']
crops = ['Rice', 'Wheat', 'Pulses', 'Cotton', 'Maize']

crop_params = {
    'Rice':    {'base_yield': 3.2, 'market_price': 18000},
    'Wheat':   {'base_yield': 3.0, 'market_price': 18500},
    'Pulses':  {'base_yield': 1.2, 'market_price': 60000},
    'Cotton':  {'base_yield': 1.5, 'market_price': 55000},
    'Maize':   {'base_yield': 2.8, 'market_price': 16500},
}

df = pd.DataFrame({
    'State': np.random.choice(states, N),
    'Crop': np.random.choice(crops, N),
    'Farm_Size': np.round(np.random.uniform(0.5, 5.0, N), 2),
    'Irrigated': np.random.choice([0, 1], size=N, p=[0.3, 0.7]),
    'Insured': np.random.choice([0, 1], size=N, p=[0.6, 0.4]),
    'Subsidy_Received': np.random.choice([0, 1], size=N, p=[0.5, 0.5]),
    'Rainfall_Index': np.round(np.random.uniform(0.5, 1.5, N), 2),
    'Temperature_Anomaly': np.round(np.random.normal(0, 1, N), 2),
    'pH_Level': np.round(np.random.uniform(5.5, 8.5, N), 2),
    'N': np.round(np.random.uniform(50, 300, N), 1),
    'P': np.round(np.random.uniform(10, 100, N), 1),
    'K': np.round(np.random.uniform(50, 300, N), 1),
    'Humidity': np.round(np.random.uniform(30, 95, N), 1),
    'Market_Access_Distance': np.round(np.random.uniform(1, 50, N), 1),
    'Technology_Score': np.round(np.random.uniform(0, 1, N), 2),
    'Age': np.random.randint(21, 70, N),
    'Education_Years': np.random.randint(0, 16, N),
})

def compute_features(row):
    crop = row['Crop']
    base_yield = crop_params[crop]['base_yield']
    market_price = crop_params[crop]['market_price']

    climate_effect = 1.0
    if row['Rainfall_Index'] < 0.8:
        climate_effect -= 0.22
    elif row['Rainfall_Index'] > 1.2:
        climate_effect -= 0.10 if crop == 'Rice' else 0.20
    climate_effect -= 0.07 * row['Temperature_Anomaly']
    if row['Irrigated']:
        climate_effect += 0.12
    climate_effect = max(0.5, climate_effect)

    yield_per_ha = base_yield * climate_effect + np.random.normal(0, 0.1)
    yield_per_ha = max(0.5, yield_per_ha)
    npk_ratio = (row['N'] + row['P'] + row['K']) / 3
    nutrient_interaction = row['N'] * row['P'] * row['K']
    land_utilization = np.clip(np.random.uniform(0.7, 1.0), 0, 1)
    price_volatility = np.random.uniform(0.03, 0.08) * market_price
    gdd = max(0, (25 + row['Temperature_Anomaly']) - 10)
    heat_index = (25 + row['Temperature_Anomaly']) + 0.5 * row['Humidity'] - 14.5
    monsoon_impact = row['Rainfall_Index'] * (1 - abs(row['Temperature_Anomaly']/4))
    pH_opt_index = np.exp(-abs(row['pH_Level'] - 6.5))
    subsidy_amt = 6000 if row['Subsidy_Received'] else 0
    income = row['Farm_Size'] * yield_per_ha * market_price * land_utilization + subsidy_amt
    if row['Insured']:
        income *= np.random.uniform(1.07, 1.15)
    yield_volatility = np.abs(np.random.normal(0.15, 0.05))

    return pd.Series({
        'Yield_t_per_ha': yield_per_ha,
        'Yield_Volatility': yield_volatility,
        'Market_Price': market_price,
        'NPK_Ratio': npk_ratio,
        'Nutrient_Interaction': nutrient_interaction,
        'Land_Utilization': land_utilization,
        'Price_Volatility': price_volatility,
        'GDD': gdd,
        'Heat_Index': heat_index,
        'Monsoon_Impact': monsoon_impact,
        'pH_Optimization_Index': pH_opt_index,
        'Annual_Income': income
    })

features_df = df.apply(compute_features, axis=1)
df = pd.concat([df, features_df], axis=1)

for i in range(1, 4):
    df[f'Prev_Income_{i}'] = df['Annual_Income'] * np.random.uniform(0.7, 1.2, N)

# 2. Preprocessing
categorical = ['State', 'Crop']
numerical = [
    'Farm_Size', 'Irrigated', 'Insured', 'Subsidy_Received', 'Rainfall_Index', 'Temperature_Anomaly',
    'pH_Level', 'N', 'P', 'K', 'Humidity', 'Market_Access_Distance',
    'Technology_Score', 'Age', 'Education_Years',
    'Yield_t_per_ha', 'Yield_Volatility', 'Market_Price', 'NPK_Ratio',
    'Nutrient_Interaction', 'Land_Utilization', 'Price_Volatility',
    'GDD', 'Heat_Index', 'Monsoon_Impact', 'pH_Optimization_Index',
    'Prev_Income_1', 'Prev_Income_2', 'Prev_Income_3'
]
target = 'Annual_Income'

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('num', StandardScaler(), numerical)
])


# 3. Train/Test Split
X = df[categorical + numerical]
y = df[target]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
rel_val = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=rel_val, random_state=42)

X_train_prep = preprocess.fit_transform(X_train)
X_val_prep = preprocess.transform(X_val)
X_test_prep = preprocess.transform(X_test)

# 4. Train Models
xgb = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb.fit(X_train_prep, y_train)
xgb_pred_val = xgb.predict(X_val_prep)

lgb = LGBMRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
lgb.fit(X_train_prep, y_train)
lgb_pred_val = lgb.predict(X_val_prep)

ensemble_inputs_val = np.vstack([xgb_pred_val, lgb_pred_val]).T
meta = RidgeCV()
meta.fit(ensemble_inputs_val, y_val)
final_pred_val = meta.predict(ensemble_inputs_val)

def show_metrics(y_true, preds, name):
    mape = mean_absolute_percentage_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    print(f"{name}: MAPE={mape*100:.2f}%, RMSE=₹{rmse:.0f}, R2={r2:.2f}")

print("\n### Validation Metrics ###")
show_metrics(y_val, xgb_pred_val, "XGBoost")
show_metrics(y_val, lgb_pred_val, "LightGBM")
show_metrics(y_val, final_pred_val, "Stacked Ensemble")

xgb_pred_test = xgb.predict(X_test_prep)
lgb_pred_test = lgb.predict(X_test_prep)
ensemble_inputs_test = np.vstack([xgb_pred_test, lgb_pred_test]).T
final_pred_test = meta.predict(ensemble_inputs_test)

print("\n### Test Set ###")
show_metrics(y_test, final_pred_test, "Stacked Ensemble (Test)")

# 5. Save Models
joblib.dump(preprocess, 'preprocess.pkl')
joblib.dump(xgb, 'xgb.pkl')
joblib.dump(lgb, 'lgb.pkl')
joblib.dump(meta, 'meta.pkl')
print("\nAll models and preprocessors saved (CatBoost & SHAP skipped).")

# 6. Plot Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
plt.title('Stacked Ensemble: Actual vs Predicted Income')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("Prediction plot saved as actual_vs_predicted.png")
