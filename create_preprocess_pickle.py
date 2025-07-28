import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from custom_transformer import CustomImputer

# Load your training dataset (update path if needed)
df = pd.read_csv("your_training_data.csv")  # Replace this with your actual training CSV

# Split columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Create preprocessing pipeline
preprocess = Pipeline(steps=[
    ("custom_imputer", CustomImputer()),
    ("transform", ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]))
])

# Fit the pipeline
preprocess.fit(df)

# Save pipeline
joblib.dump(preprocess, "preprocess.pkl")
print("✅ Saved preprocess.pkl successfully.")
