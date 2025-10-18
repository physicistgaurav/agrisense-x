import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_classes = sorted([
    d for d in __import__("os").listdir("data/plantvillage")
])

model = mobilenet_v2()
model.classifier[1] = torch.nn.Linear(1280, len(dataset_classes))
model.load_state_dict(torch.load("model/model.pth", map_location=DEVICE))
model.eval()


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    return dataset_classes[idx], float(conf)
