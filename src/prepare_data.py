from datasets import load_dataset
import os

ds = load_dataset("saakshigupta/deepfake-detection-dataset-v3")

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

limit = 150   # keep small for now

for i, item in enumerate(ds["train"]):
    if i >= limit:
        break

    img = item["image"]
    label = item["label"]

    if label == 1:
        img.save(f"data/real/{i}.jpg")
    else:
        img.save(f"data/fake/{i}.jpg")

print("Done")
