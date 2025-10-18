import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing image for 224 x 224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# load dataset => around 20K images total
dataset = ImageFolder("data/plantvillage", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = len(dataset.classes)

# transfer learning through movileNetV2
model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, num_classes)
model = model.to(DEVICE)

# loss function and learning rate optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model/model.pth")
print("Model saved")
