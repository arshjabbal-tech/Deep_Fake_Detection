import pandas as pd
import matplotlib.pyplot as plt

# Example results (you will replace later dynamically)
data = [
    ["M1 (ResNet)", 0.85, 0.80, 0.82, 0.85, 0.83, 0.84],
    ["M2 (MobileNet)", 0.82, 0.78, 0.80, 0.82, 0.81, 0.82],
    ["M3 (EfficientNet)", 0.88, 0.83, 0.85, 0.88, 0.86, 0.87],
    ["M4", 0,0,0,0,0,0],
    ["M5", 0,0,0,0,0,0],
    ["M6", 0,0,0,0,0,0],
    ["M7", 0,0,0,0,0,0],
    ["M8", 0,0,0,0,0,0],
    ["M9", 0,0,0,0,0,0],
    ["Ensemble", 0,0,0,0,0,0],
]

columns = ["Models", "Sensitivity", "Specificity", "Precision", "Recall", "F1", "Accuracy"]

df = pd.DataFrame(data, columns=columns)

# Create figure
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Save image
plt.savefig("results/final_table.png", bbox_inches='tight')
plt.show()