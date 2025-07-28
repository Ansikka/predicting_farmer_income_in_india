🌾 Farmer Income Prediction in India
A real-world machine learning project built to predict the income of Indian farmers using demographic, agricultural, 
and socio-economic data. This app provides insights that could help in policy design, financial planning, and risk assessment for rural households.

🔍 Overview
Indian farmers face volatile income due to unpredictable weather, fluctuating market prices, and limited access to modern agricultural infrastructure. This Streamlit-powered web app leverages machine learning to predict a farmer's annual income based on several input parameters — giving decision-makers a data-driven perspective on rural economics.

🛠 Tech Stack
🐍 Python 3.12+

🧠 Scikit-learn

📦 Joblib

📊 Pandas & NumPy

🌐 Streamlit (for UI)

🧪 Custom Transformers

☁️ Deployed on: Streamlit Cloud

⚙️ Features
✅ Upload farmer data (CSV) for batch prediction
✅ View predicted income instantly
✅ Clean, responsive UI with contextual insights
✅ Custom-built preprocessing pipeline with CustomImputer
✅ Re-trainable model architecture (optional)
✅ Lightweight and deployable under GitHub’s file size limits

📁 Project Structure
bash
Copy
Edit
📦 project/
├── app.py                    # Streamlit web app

├── custom_transformer.py     # Custom imputer class

├── create_preprocess_pickle.py # Script to regenerate preprocess.pkl

├── preprocess.pkl            # Saved preprocessing pipeline

├── farmer_income_model.pkl   # Trained ML model (joblib format)

├── your_training_data.csv    # Original training data

└── README.md                 # You are here

🚀 Quick Start
📌 Prerequisites
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should include:

nginx
Copy
Edit
streamlit
scikit-learn
pandas
numpy
joblib
▶️ Run Locally
bash
Copy
Edit
streamlit run app.py
🌐 Deploy on Streamlit Cloud
Push your repo to GitHub.

Go to Streamlit Cloud and click “New app”.

Connect your repo and select app.py as the entry point.

Deploy 🎉

🔄 How to Regenerate preprocess.pkl?
If you modify or retrain the model, regenerate the pipeline by running:

bash
Copy
Edit
python create_preprocess_pickle.py
Make sure custom_transformer.py is in the same folder.

📉 Sample Inputs (for Testing)
State: Bihar

Crop Type: Rice

Area: 2.5 acres

Rainfall: 1200 mm

Fertilizer Usage: High

Market Access: Poor

Govt Subsidy: No

Result: Income ~ ₹85,000/year

 Motivation
This app is more than a model — it’s a statement that data can be inclusive, even for the most underserved communities.

Agricultural incomes in India are affected by a complex web of local and national factors. By building tools like this, we move closer to democratizing data-driven insights for public policy, non-profits, and rural fintech startups.

🙌 Acknowledgements
Built as part of the "Predicting Farmer Income in India" competition on Unstop.

Dataset: Synthetic + Real-world inspired fields

Thanks to the open-source community for scikit-learn and Streamlit.

📫 Contact
Anshika Sharma
