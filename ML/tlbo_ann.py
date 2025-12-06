# tlbo_ann.py
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
# TLBO Hyperparameter optimization (simplified)
# -----------------------------
def tlbo_optimization(pop_size=10, generations=20):
    best_score = -np.inf
    best_params = None
    for _ in range(generations):
        for _ in range(pop_size):
            hidden_layer1 = np.random.randint(5, 20)
            hidden_layer2 = np.random.randint(5, 20)
            mlp = MLPRegressor(hidden_layer_sizes=(hidden_layer1, hidden_layer2),
                               max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)
            score = r2_score(y_test, mlp.predict(X_test))
            if score > best_score:
                best_score = score
                best_params = (hidden_layer1, hidden_layer2)
    return best_params, best_score

best_layers, best_r2 = tlbo_optimization()
print(f"Best hidden layers: {best_layers}, Best R² (during optimization): {best_r2*100:.2f}%")

# Train final model with best hyperparameters
mlp_final = MLPRegressor(hidden_layer_sizes=best_layers, max_iter=500, random_state=42)
mlp_final.fit(X_train, y_train)
y_pred = mlp_final.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"TLBO-ANN Final Model Evaluation:")
print(f"R²: {r2*100:.2f}%")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# -----------------------------
# K-Fold Cross Validation (to check overfitting)
# -----------------------------
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer

# Define 5-Fold Cross Validation
k = 9
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define scoring metrics
r2_scorer = make_scorer(r2_score)
mse_scorer = make_scorer(mean_squared_error)

# R² scores
r2_scores = cross_val_score(
    mlp_final, X_scaled, y, cv=kf, scoring=r2_scorer
)

# MSE scores
mse_scores = -cross_val_score(
    mlp_final, X_scaled, y, cv=kf, scoring='neg_mean_squared_error'
)

# RMSE (square root of each MSE)
rmse_scores = np.sqrt(mse_scores)

print("\nK-Fold Cross Validation Results (TLBO-ANN):")
print(f"Average R²: {np.mean(r2_scores)*100:.2f}% ± {np.std(r2_scores)*100:.2f}%")
print(f"Average MSE: {np.mean(mse_scores):.6f}")
print(f"Average RMSE: {np.mean(rmse_scores):.6f}")
