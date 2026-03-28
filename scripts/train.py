import torch
from src.models.resnet_adapter import ResNet50WithConvParallelAdapters
from src.data.dataloaders import get_dataloaders
from src.training.trainer import trainer
from src.utils.seed import set_seed, seed_worker

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    g = torch.Generator().manual_seed(42)

    train_dl, val_dl, class_to_idx = get_dataloaders(
        data_dir="/data/flower_data",
        batch_size=64,
        seed_worker=seed_worker,
        generator=g
    )

    model = ResNet50WithConvParallelAdapters(num_classes=len(class_to_idx)).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, momentum=0.9, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2
    )

    trainer(
        train_dl, val_dl, model,
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        optimizer=optimizer,
        epochs=30,
        early_stop_patience=5,
        device=device,
        scheduler=scheduler,
        scheduler_type="plateau"
    )

if __name__ == "__main__":
    main()
