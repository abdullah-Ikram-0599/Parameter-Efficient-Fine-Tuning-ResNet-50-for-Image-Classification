import gradio as gr
import torch
from pathlib import Path
from torchvision.datasets import ImageFolder
import sys
import os
import json
from src.inference.predict import (
    load_model,
    predict_image,
    get_eval_transform,
    load_label_mapping
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_TO_IDX_JSON = PROJECT_ROOT / "src" / "inference" / "class_to_idx.json"
CAT_TO_NAME_JSON  = PROJECT_ROOT / "src" / "inference" / "cat_to_name.json"

cat_to_name = load_label_mapping(CAT_TO_NAME_JSON)

DATA_DIR = "data/train"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet50_low_rank_adapter.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



with open(CLASS_TO_IDX_JSON, "r") as f:
    class_to_idx = json.load(f)


idx_to_class = {i: str(i+1) for i in range(102)}

model = load_model(
    checkpoint_path=CHECKPOINT_PATH,
    num_classes=len(class_to_idx),
    device=DEVICE
)

transform = get_eval_transform()


def gradio_predict(image):
    class_name, confidence = predict_image(
        image=image,
        model=model,
        transform=transform,
        device=DEVICE,
        idx_to_class=idx_to_class,
        cat_to_name=cat_to_name
    )

    return (
        f"Flower Type: {class_name}\n"
    
        f"Confidence: {confidence:.2%}"
    )

outputs=gr.Textbox(
    label="Prediction",
    lines=4,          
    max_lines=10
)

interface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Flower Image"),
    outputs=outputs,
    title="Parameter-Efficient Fine-Tuning for Image Classification with ResNet-50 Conv Adapters",
    
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=3434)
