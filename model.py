import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv("./data/housing_price_dataset_clean.csv")

# Feature engineering
df['house_age'] = 2025 - df['YearBuilt']  # Add house age as a feature
df['Price_per_SquareFeet'] = df['Price'] / df['SquareFeet']  # Price per square foot

# Define features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# One-hot encode categorical features
categorical_features = ["Neighborhood"]
column_transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(drop="first"), categorical_features)
    ],
    remainder="passthrough"
)

X_encoded = column_transformer.fit_transform(X)

# Convert encoded features to DataFrame
ohe_cols = column_transformer.named_transformers_["ohe"].get_feature_names_out(categorical_features)
num_cols = [c for c in X.columns if c not in categorical_features]
X_encoded = pd.DataFrame(X_encoded, columns=list(ohe_cols) + num_cols)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,       # 20% test data
    random_state=42      # For reproducibility
)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"]
}

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_grid,  
    n_iter=100,            # Increased iterations for better search
    cv=5,                  # 5-fold cross-validation
    scoring='r2',          # Optimize for R²
    n_jobs=-1,             # Use all available cores
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best R² (CV):", random_search.best_score_)

# Train the best model
best_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

best_rf.fit(X_train, y_train)
print("Best RF model retrained")

# Evaluate the model
y_pred_best = best_rf.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print("Final Test Performance:")
print(f"MAE: {mae_best:.2f}")
print(f"MSE: {mse_best:.2f}")
print(f"RMSE: {rmse_best:.2f}")
print(f"R²: {r2_best:.2f}")

# Residuals plot
residuals = y_test - y_pred_best

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, color='teal')
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--')
plt.savefig("./residuals_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Actual vs Predicted Prices plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.savefig("./actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.show()

# Save the best model
joblib.dump(best_rf, "final_model.pkl", compress=("lzma", 9))  # Use lzma with maximum compression
print("Model saved as final_model.pkl")
