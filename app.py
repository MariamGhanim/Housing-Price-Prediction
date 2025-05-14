import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the cleaned dataset and model
data_path = "./data/housing_price_dataset_clean.csv"
model_path = "final_model.pkl"

df = pd.read_csv(data_path)
model = joblib.load(model_path)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Performance"])

# Prediction Page
if page == "Prediction":
    st.title("Housing Price Prediction")
    st.write("Enter the features below to predict the house price.")
    st.info("**Note:** The dataset used to train this model is not real, and the model achieved an R² score of 0.99.")

    # Input fields for user inputs
    st.markdown("### Input Features")
    Neighborhood = st.selectbox("🏘️ Neighborhood", df["Neighborhood"].unique())
    YearBuilt = st.number_input("📅 Year Built", min_value=1800, max_value=2050, value=2000)
    SquareFeet = st.number_input("📏 Square Feet", min_value=500, max_value=4000, value=1500)
    Bedrooms = st.slider("🛏️ Number of Bedrooms", min_value=1, max_value=5, value=3)
    Bathrooms = st.slider("🛁 Number of Bathrooms", min_value=1, max_value=4, value=2)

    # Create a DataFrame from user inputs
    input_data = {
        "Neighborhood": Neighborhood,
        "YearBuilt": YearBuilt,
        "SquareFeet": SquareFeet,
        "Bedrooms": Bedrooms,
        "Bathrooms": Bathrooms,
    }
    input_df = pd.DataFrame(input_data, index=[0])

    # Preprocess inputs
    input_df["house_age"] = 2025 - input_df["YearBuilt"]  # Adjusted for 2025 as the reference year
    input_df = input_df.drop("YearBuilt", axis=1)

    # One-hot encode categorical features
    categorical_features = ["Neighborhood"]
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

    # Align input with model features
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Prediction button
    if st.button("🔮 Predict"):
        prediction = model.predict(input_encoded)
        st.success(f"💰 Predicted House Price: **${prediction[0]:,.2f}**")

# Model Performance Page
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write("The model achieved an **R² score of 0.99**, indicating excellent performance. Below are the visualizations to support this claim.")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Display residuals plot in the first column
    with col1:
        st.subheader("Residuals Plot")
        st.image("./residuals_plot.png", caption="Residuals Plot", use_container_width=True)

    # Display actual vs predicted plot in the second column
    with col2:
        st.subheader("Actual vs Predicted Prices")
        st.image("./actual_vs_predicted.png", caption="Actual vs Predicted Prices", use_container_width=True)

    # Add a summary section
    st.markdown("### Summary")
    st.write(
        """
        - The **Residuals Plot** shows that the residuals are centered around zero, indicating good predictions.
        - The **Actual vs Predicted Prices Plot** demonstrates that the model's predictions closely match the actual prices.
        """
    )