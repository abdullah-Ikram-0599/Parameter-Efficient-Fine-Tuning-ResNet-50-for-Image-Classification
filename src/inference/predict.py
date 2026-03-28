import json
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
from src.models.resnet_adapter import ResNet50WithConvParallelAdapters
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_JSON = PROJECT_ROOT /"src"/ "inference" / "class_to_idx.json"


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet50_low_rank_adapter.pth"

with open(LABEL_JSON, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {int(v): k for k, v in class_to_idx.items()}

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_label_mapping(json_path):
   
    with open(json_path, "r") as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_model(checkpoint_path, num_classes, device):
    model = ResNet50WithConvParallelAdapters(num_classes=num_classes)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    image: Image.Image,
    model,
    transform,
    device,
    idx_to_class,
    cat_to_name
):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

    oxford_class_id = idx_to_class[pred_idx]        
    flower_name = cat_to_name[oxford_class_id]     

    return flower_name, confidence

