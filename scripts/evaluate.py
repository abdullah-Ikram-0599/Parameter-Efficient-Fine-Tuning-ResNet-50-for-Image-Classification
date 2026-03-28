import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.utils.seed import set_seed
from src.models.resnet_adapter import ResNet50WithConvParallelAdapters

def get_val_transform():
    """Deterministic preprocessing for validation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def evaluate(model, dataloader, device):
    """Compute top-1 accuracy on validation set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def main(args):
  
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = datasets.ImageFolder(
        root=args.val_dir,
        transform=get_val_transform()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = ResNet50WithConvParallelAdapters(num_classes=len(val_dataset.classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    accuracy = evaluate(model, val_loader, device)

    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet50 + Parallel Adapters on Validation Set"
    )

    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to validation dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    main(args)
    
