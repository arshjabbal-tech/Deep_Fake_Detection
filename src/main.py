import os
import pandas as pd
import matplotlib.pyplot as plt

from models import *
from train import train_model
from evaluate import compute_metrics


# ✅ LOAD DATA
def load_data():
    X = []
    y = []

    for label, folder in enumerate(["fake", "real"]):
        path = f"data/{folder}"

        for file in os.listdir(path):
            X.append(os.path.join(path, file))
            y.append(label)

    return X, y


# ✅ RUN ALL MODELS
def run_all_models():
    X, y = load_data()   # ✅ y is defined here properly

    results = {}

    models_list = {
        "M1 (ResNet18)": load_resnet18(),
        "M2 (MobileNet)": load_mobilenet(),
        "M3 (EfficientNet)": load_efficientnet(),
        "M4 (VGG16)": load_vgg(),
        "M5 (DenseNet)": load_densenet(),
        "M6 (AlexNet)": load_alexnet(),
        "M7 (SqueezeNet)": load_squeezenet(),
        "M8 (ShuffleNet)": load_shufflenet(),
        "M9 (ResNet34)": load_resnet34(),
        "M10 (GoogLeNet)": load_googlenet(),
    }

    for name, model in models_list.items():
        print(f"\nRunning {name}...")

        # ✅ train each model
        model = train_model(model, X, y)

        # ✅ predict
        y_pred = []
        for img_path in X:
            pred = predict(model, img_path)
            y_pred.append(pred)

        # ✅ compute metrics
        results[name] = compute_metrics(y, y_pred)

    return results


# ✅ MAIN
if __name__ == "__main__":
    results = run_all_models()

    print("\nFINAL RESULTS:\n")

    data = []
    for model, vals in results.items():
        data.append([model] + list(vals))

    columns = ["Model", "Sensitivity", "Specificity", "Precision", "Recall", "F1", "Accuracy"]

    df = pd.DataFrame(data, columns=columns)

    # ✅ round to 3 decimals
    df = df.round(3)

    print(df)

    # ✅ save table as image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.savefig("final_table.png")
    print("\nTable saved as final_table.png")