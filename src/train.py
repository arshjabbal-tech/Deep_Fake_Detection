import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def train_model(model, X, y, epochs=2):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0

        for img_path, label in zip(X, y):
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)

            optimizer.zero_grad()
            outputs = model(img)
            
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model