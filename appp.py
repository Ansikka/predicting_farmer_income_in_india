
''''
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load or define global variables
model = None
preprocessor = None
model_path = "farmer_income_model.pkl"

# Define categorical and numerical features
categorical = ['Gender', 'State_Name', 'District_Name', 'Season', 'Crop']
numerical = ['Age', 'Annual_Rainfall', 'Temperature', 'Humidity', 'Fertilizer_Amount', 'Pesticide_Amount']

def train_model(df, model_type='xgboost'):
    global preprocessor

    X = df[categorical + numerical]
    y = df['Income']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('num', StandardScaler(), numerical)
    ])

    if model_type == 'xgboost':
        regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        regressor = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', regressor)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, model_path)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

def load_model():
    global model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return True
    return False

def predict(df):
    global model
    return model.predict(df)

# ----------- Streamlit UI ---------------- #
st.title("🌾 Farmer Income Prediction")

st.sidebar.header("Options")
option = st.sidebar.radio("Choose an action:", ["Upload & Predict", "Retrain Model"])

if option == "Upload & Predict":
    uploaded_file = st.file_uploader("Upload farmer data (.csv)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", data)

        if load_model():
            try:
                input_data = data[categorical + numerical]
                prediction = predict(input_data)
                st.success("✅ Prediction Complete")
                st.write("📊 Predicted Income:")
                st.dataframe(pd.DataFrame({"Predicted Income": prediction}))
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.warning("⚠️ No trained model found. Please retrain first.")

elif option == "Retrain Model":
    uploaded_file = st.file_uploader("Upload training data with 'Income' column (.csv)", type=["csv"])
    model_type = st.selectbox("Choose Model", ["xgboost", "lightgbm"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Income' not in df.columns:
            st.error("❌ 'Income' column not found. Cannot train.")
        else:
            with st.spinner("Training model..."):
                metrics = train_model(df, model_type=model_type)
                st.success("✅ Model Trained and Saved")
                st.write("📈 Model Performance on Test Set:")
                st.json(metrics)
'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from custom_transformer import CustomImputer  # or whatever the class name is

# Load models
xgb = joblib.load("xgb.pkl")
lgb = joblib.load("lgb.pkl")
meta = joblib.load("meta.pkl")
preprocess = joblib.load("preprocess.pkl")

st.set_page_config(page_title="Farmer Income Predictor")
st.title("🌾 Farmer Income Predictor")

st.write("Enter farmer details manually or upload a CSV file for prediction.")

# Input from CSV
uploaded_file = st.file_uploader("📁 Upload CSV with Farmer Data", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data:")
    st.dataframe(input_df)

    if st.button("🔍 Predict from CSV"):
        X_prep = preprocess.transform(input_df)
        xgb_pred = xgb.predict(X_prep)
        lgb_pred = lgb.predict(X_prep)
        ensemble_input = np.vstack([xgb_pred, lgb_pred]).T
        final_pred = meta.predict(ensemble_input)
        input_df['Predicted_Income'] = final_pred
        st.success("✅ Predictions completed.")
        st.dataframe(input_df)
        st.download_button("⬇️ Download Results", input_df.to_csv(index=False), "predicted_incomes.csv")
else:
    st.subheader("Manual Input")

    states = ['Punjab', 'Maharashtra', 'UP', 'Bihar', 'Tamil Nadu', 'MP', 'Rajasthan']
    crops = ['Rice', 'Wheat', 'Pulses', 'Cotton', 'Maize']

    with st.form("farmer_form"):
        state = st.selectbox("State", states)
        crop = st.selectbox("Crop", crops)
        farm_size = st.number_input("Farm Size (acres)", 0.1, 10.0, 1.0)
        irrigated = st.selectbox("Irrigated", [0, 1])
        insured = st.selectbox("Insured", [0, 1])
        subsidy = st.selectbox("Subsidy Received", [0, 1])
        rainfall = st.slider("Rainfall Index", 0.5, 1.5, 1.0)
        temp_anom = st.slider("Temperature Anomaly", -2.0, 2.0, 0.0)
        ph = st.slider("pH Level", 5.5, 8.5, 6.5)
        N = st.slider("Nitrogen (N)", 50, 300, 100)
        P = st.slider("Phosphorus (P)", 10, 100, 50)
        K = st.slider("Potassium (K)", 50, 300, 100)
        humidity = st.slider("Humidity (%)", 30, 95, 60)
        distance = st.slider("Market Distance (km)", 1, 50, 10)
        tech = st.slider("Technology Score", 0.0, 1.0, 0.5)
        age = st.slider("Farmer Age", 21, 70, 40)
        edu = st.slider("Education Years", 0, 16, 8)
        prev1 = st.number_input("Previous Income 1", 0.0, 1e6, 50000.0)
        prev2 = st.number_input("Previous Income 2", 0.0, 1e6, 52000.0)
        prev3 = st.number_input("Previous Income 3", 0.0, 1e6, 53000.0)

        submit = st.form_submit_button("🔍 Predict Income")

    if submit:
        input_df = pd.DataFrame([{
            'State': state,
            'Crop': crop,
            'Farm_Size': farm_size,
            'Irrigated': irrigated,
            'Insured': insured,
            'Subsidy_Received': subsidy,
            'Rainfall_Index': rainfall,
            'Temperature_Anomaly': temp_anom,
            'pH_Level': ph,
            'N': N,
            'P': P,
            'K': K,
            'Humidity': humidity,
            'Market_Access_Distance': distance,
            'Technology_Score': tech,
            'Age': age,
            'Education_Years': edu,
            'Yield_t_per_ha': 1.5,  # dummy
            'Yield_Volatility': 0.1,
            'Market_Price': 20000,
            'NPK_Ratio': (N+P+K)/3,
            'Nutrient_Interaction': N*P*K,
            'Land_Utilization': 0.9,
            'Price_Volatility': 0.05 * 20000,
            'GDD': max(0, (25 + temp_anom) - 10),
            'Heat_Index': (25 + temp_anom) + 0.5 * humidity - 14.5,
            'Monsoon_Impact': rainfall * (1 - abs(temp_anom/4)),
            'pH_Optimization_Index': np.exp(-abs(ph - 6.5)),
            'Prev_Income_1': prev1,
            'Prev_Income_2': prev2,
            'Prev_Income_3': prev3
        }])

        X_prep = preprocess.transform(input_df)
        xgb_pred = xgb.predict(X_prep)
        lgb_pred = lgb.predict(X_prep)
        ensemble_input = np.vstack([xgb_pred, lgb_pred]).T
        final_pred = meta.predict(ensemble_input)

        st.success(f"✅ Predicted Annual Income: ₹ {final_pred[0]:,.2f}")
