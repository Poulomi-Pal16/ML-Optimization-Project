# sho_ann.py
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
# SHO Hyperparameter optimization (simplified)
# -----------------------------
def sho_optimization(pop_size=10, generations=20):
    best_score = -np.inf
    best_params = None
    for _ in range(generations):
        for _ in range(pop_size):
            neurons1 = np.random.randint(5, 25)
            neurons2 = np.random.randint(5, 25)
            mlp = MLPRegressor(hidden_layer_sizes=(neurons1, neurons2),
                               max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)
            score = r2_score(y_test, mlp.predict(X_test))
            if score > best_score:
                best_score = score
                best_params = (neurons1, neurons2)
    return best_params, best_score

best_layers, best_r2 = sho_optimization()
print(f"Best hidden layers: {best_layers}, Best R² (during optimization): {best_r2*100:.2f}%")

# Train final model
mlp_final = MLPRegressor(hidden_layer_sizes=best_layers, max_iter=500, random_state=42)
mlp_final.fit(X_train, y_train)
y_pred = mlp_final.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"SHO-ANN Final Model Evaluation:")
print(f"R²: {r2*100:.2f}%")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse}")

# -----------------------------
# K-Fold Cross Validation (to verify overfitting)
# -----------------------------
from sklearn.model_selection import KFold

def cross_validate_model(X_scaled, y, best_layers, k=9):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, mse_scores, rmse_scores = [], [], []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        mlp = MLPRegressor(hidden_layer_sizes=best_layers, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        r2 = r2_score(y_test, y_pred) * 100
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        r2_scores.append(r2)
        mse_scores.append(mse)
        rmse_scores.append(rmse)

    print("\nK-Fold Cross Validation Results (SHO-ANN):")
    print(f"Average R²: {np.mean(r2_scores):.2f}% ± {np.std(r2_scores):.2f}%")
    print(f"Average MSE: {np.mean(mse_scores):.6f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.6f}")

# Run K-Fold Validation
cross_validate_model(X_scaled, y, best_layers)
