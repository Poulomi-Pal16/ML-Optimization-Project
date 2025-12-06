import matplotlib.pyplot as plt
import numpy as np

# Updated evaluation metrics
models = ['TLBO-ANN', 'SHO-ANN', 'HHO-MLP']
r2_scores = [99.96, 99.98, 99.97]
mse_values = [0.000175, 0.000095, 0.000107]
rmse_values = [0.01324, 0.00976, 0.01034]
x = np.arange(len(models))
width = 0.25

# Dual-axis bar chart
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis (R²)
ax1.bar(x - width, r2_scores, width, label='R² (%)', color='gold')
ax1.set_ylabel('R² (%)', color='gold')
ax1.tick_params(axis='y', labelcolor='gold')
ax1.set_ylim(99.90, 100.00)

# Right axis (MSE / RMSE)
ax2 = ax1.twinx()
ax2.bar(x, mse_values, width, label='MSE', color='skyblue')
ax2.bar(x + width, rmse_values, width, label='RMSE', color='seagreen')
ax2.set_ylabel('MSE / RMSE', color='teal')
ax2.tick_params(axis='y', labelcolor='teal')

# Common formatting
plt.title("Performance Comparison of TLBO-ANN, SHO-ANN, and HHO-MLP")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper right')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))

plt.tight_layout()
plt.savefig("Updated_ANN_Comparison_DualAxis.png", dpi=300)
plt.show()
