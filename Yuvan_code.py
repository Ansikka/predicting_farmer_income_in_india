
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# Load models and preprocessor
preprocess = joblib.load("preprocess.pkl")
xgb = joblib.load("xgb.pkl")
lgb = joblib.load("lgb.pkl")
meta = joblib.load("meta.pkl")

# Streamlit page config for showing result and website
st.set_page_config(page_title="Farmer Income Predictor")
st.title("🌾 Farmer Income Predictor")
st.write("Enter farmer details manually or upload a CSV file for prediction.")

# Expected columns for preprocessing
expected_columns = [
    'Farm_Size', 'Irrigated', 'Insured', 'Subsidy_Received', 'Rainfall_Index',
    'Temperature_Anomaly', 'pH_Level', 'N', 'P', 'K', 'Humidity',
    'Market_Access_Distance', 'Technology_Score', 'Age', 'Education_Years',
    'Yield_t_per_ha', 'Yield_Volatility', 'Market_Price', 'NPK_Ratio',
    'Nutrient_Interaction', 'Land_Utilization', 'Price_Volatility', 'GDD',
    'Heat_Index', 'Monsoon_Impact', 'pH_Optimization_Index',
    'Prev_Income_1', 'Prev_Income_2', 'Prev_Income_3', 'State', 'Crop'
]

# --- CSV Upload Section ---
uploaded_file = st.file_uploader("📁 Upload CSV with Farmer Data", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(input_df)

    if st.button("🔍 Predict from CSV"):
        # Fill missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 'Unknown' if col in ['State', 'Crop'] else 0
        input_df = input_df[expected_columns]

        try:
            # Debug check
            loaded_columns = preprocess.feature_names_in_
            missing_cols = set(loaded_columns) - set(input_df.columns)
            extra_cols = set(input_df.columns) - set(loaded_columns)
            if missing_cols:
                st.error(f"❌ Missing columns in CSV: {missing_cols}")
            if extra_cols:
                st.warning(f"⚠️ Extra columns in CSV: {extra_cols}")

            # Predict
            X_prep = preprocess.transform(input_df)
            xgb_pred = xgb.predict(X_prep)
            lgb_pred = lgb.predict(X_prep)
            ensemble_input = np.vstack([xgb_pred, lgb_pred]).T
            final_pred = meta.predict(ensemble_input)

            input_df['Predicted_Annual_Income'] = final_pred.astype(int)
            st.success("✅ Prediction completed.")
            st.dataframe(input_df[['State', 'Crop', 'Farm_Size', 'Predicted_Annual_Income']])
            st.download_button("⬇️ Download Results", input_df.to_csv(index=False), "predicted_incomes.csv")
        except Exception as e:
            st.error(f"❌ Prediction Failed: {str(e)}")

# --- Manual Input Section ---

else:
    st.subheader("📝 Manual Input")

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
            'Yield_t_per_ha': 1.5,
            'Yield_Volatility': 0.1,
            'Market_Price': 20000,
            'NPK_Ratio': (N + P + K) / 3,
            'Nutrient_Interaction': N * P * K,
            'Land_Utilization': 0.9,
            'Price_Volatility': 0.05 * 20000,
            'GDD': max(0, (25 + temp_anom) - 10),
            'Heat_Index': (25 + temp_anom) + 0.5 * humidity - 14.5,
            'Monsoon_Impact': rainfall * (1 - abs(temp_anom / 4)),
            'pH_Optimization_Index': np.exp(-abs(ph - 6.5)),
            'Prev_Income_1': prev1,
            'Prev_Income_2': prev2,
            'Prev_Income_3': prev3
        }])

        try:
            X_prep = preprocess.transform(input_df)
            xgb_pred = xgb.predict(X_prep)
            lgb_pred = lgb.predict(X_prep)
            ensemble_input = np.vstack([xgb_pred, lgb_pred]).T
            final_pred = meta.predict(ensemble_input)

            st.success(f"✅ Predicted Annual Income: ₹ {final_pred[0]:,.2f}")
        except Exception as e:
            st.error(f"❌ Prediction Failed: {str(e)}")

    #            ****************** streaming on STREAMLIT APP ***********************

    st.subheader("📈 Income Distribution of Farmers")

# Set income threshold
income_threshold = 100000

# Only plot if prediction column exists and has multiple entries
if 'input_df' in locals() and 'Predicted_Annual_Income' in input_df.columns and len(input_df) > 1:
    # Create color list
    colors = ['red' if income < income_threshold else 'green' for income in input_df['Predicted_Annual_Income']]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(input_df)), input_df['Predicted_Annual_Income'], color=colors)
    ax.axhline(y=income_threshold, color='blue', linestyle='--', label=f'Threshold: ₹{income_threshold}')
    ax.set_title("Predicted Income per Farmer")
    ax.set_xlabel("Farmer Index")
    ax.set_ylabel("Predicted Annual Income (₹)")
    ax.legend()

    st.pyplot(fig)
elif 'Predicted_Annual_Income' in locals():
    st.info("📌 Only one prediction available. Bar chart is shown for multiple predictions.")
else:
    st.warning("⚠️ No predicted income data available to plot.")
