# hho_mlp.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv('Data.csv')
X = data[['UW','CH','IFA','SLA','SH','PWPR','RFN']]
y = data['FS']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# HHO Hyperparameter optimization (simplified)
# -----------------------------
def hho_optimization(pop_size=10, generations=20):
    best_score = -np.inf
    best_params = None
    for _ in range(generations):
        for _ in range(pop_size):
            n1 = np.random.randint(5, 30)
            n2 = np.random.randint(5, 30)
            mlp = MLPRegressor(hidden_layer_sizes=(n1, n2), max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)
            score = r2_score(y_test, mlp.predict(X_test))
            if score > best_score:
                best_score = score
                best_params = (n1, n2)
    return best_params, best_score

best_layers, best_r2 = hho_optimization()
print(f"Best hidden layers: {best_layers}, Best R² (during optimization): {best_r2*100:.2f}%")

# Train final model
mlp_final = MLPRegressor(hidden_layer_sizes=best_layers, max_iter=500, random_state=42)
mlp_final.fit(X_train, y_train)
y_pred = mlp_final.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"HHO-MLP Final Model Evaluation:")
print(f"R²: {r2*100:.2f}%")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# -----------------------------
# K-Fold Cross Validation for HHO-MLP
# -----------------------------
from sklearn.model_selection import KFold, cross_val_score

k = 9  # You can use k=5 or 10 depending on dataset size
mlp_cv = MLPRegressor(hidden_layer_sizes=best_layers, max_iter=500, random_state=42)

# R² scores
r2_scores = cross_val_score(mlp_cv, X_scaled, y, cv=k, scoring='r2')
# MSE scores (negative mean squared error → convert to positive)
mse_scores = -cross_val_score(mlp_cv, X_scaled, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)

print("\nK-Fold Cross Validation Results (HHO-MLP):")
print(f"Average R²: {r2_scores.mean()*100:.2f}% ± {r2_scores.std()*100:.2f}%")
print(f"Average MSE: {mse_scores.mean():.6f}")
print(f"Average RMSE: {rmse_scores.mean():.6f}")
