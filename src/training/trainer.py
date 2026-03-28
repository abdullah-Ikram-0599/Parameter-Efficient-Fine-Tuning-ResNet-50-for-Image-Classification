import copy
import torch

def freeze_backbone_bn(model):
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
def trainer(train_dataloader,  val_dataloader, model, loss_fn, optimizer, epochs, early_stop_patience, device, 
            scheduler=None, scheduler_type=None, max_norm=1.0
):
    no_improve_epochs = 0

    train_loss = []
    
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    model = model.to(device)

    for epoch in range(epochs):

        train_batch_loss = 0.0
        val_batch_loss = 0.0

        train_total = 0
        val_total = 0

        train_correct = 0
        val_correct = 0

        model.train()
        freeze_backbone_bn(model)

        for X_train, y_train in train_dataloader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(X_train)
            loss = loss_fn(pred, y_train)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=max_norm
            )

            optimizer.step()

            pred_labels = pred.argmax(dim=1)

            train_batch_loss += loss.item() * y_train.size(0)
            train_total += y_train.size(0)
            train_correct += (pred_labels == y_train).sum().item()

        train_loss.append(train_batch_loss / max(train_total, 1))
        train_accuracy.append(train_correct / max(train_total, 1))

     
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                pred = model(X_val)
                loss = loss_fn(pred, y_val)

                val_batch_loss += loss.item() * y_val.size(0)

                pred_labels = pred.argmax(dim=1)
                val_correct += (pred_labels == y_val).sum().item()
                val_total += y_val.size(0)

        val_loss.append(val_batch_loss / max(val_total, 1))
        val_accuracy.append(val_correct / max(val_total, 1))

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Train Loss: {train_loss[-1]:.4f}, "
            f"Train Acc: {train_accuracy[-1]:.4f} | "
            f"Val Loss: {val_loss[-1]:.4f}, "
            f"Val Acc: {val_accuracy[-1]:.4f}"
        )

      
        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "resnet50_adapter_only.pth")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

     
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss[-1])
            elif scheduler_type == "epoch":
                scheduler.step()
            else:
                raise ValueError("scheduler_type must be 'plateau' or 'epoch'")

        if no_improve_epochs >= early_stop_patience:
            print(f"Stopped early at epoch: {epoch+1}")
            break

    model.load_state_dict(best_state)

    result = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }

    return result
