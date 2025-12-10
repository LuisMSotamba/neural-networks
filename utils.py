import math
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

def show_images_in_grid(
    tensor_images,
    labels_images,
    cols=3,
    figsize=(12, 8)
):
    """
    Display images in a grid layout.

    Args:
        tensor_images (tensor): A tensor object.
        cols (int): Number of columns in the grid.
        figsize (tuple): Size of the figure.
    """
    n_images = len(tensor_images)
    rows = math.ceil(n_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (ax, img_tensor) in enumerate(zip(axes, tensor_images)):
        try:
            img = img_tensor.clone()
            img = img.permute(1,2,0)
            img = img.numpy()
            ax.imshow(img)
            ax.set_title(i, fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")

    # Hide empty subplots if any
    for ax in axes[n_images:]:
        ax.axis("off")
    plt.title(labels_images[0])
    plt.tight_layout()
    plt.show()


def get_weight_vector(class_value, device="cuda"):
    import warnings
    warnings.warn(f"Be sure encoder.classes_ and class_value.keys() are the same")
    # Suppose you have 4 classes
    num_classes = len(class_value.keys())
    
    # Example: these could come from your dataset counts
    class_counts = torch.tensor(list(class_value.values()), dtype=torch.float).to(device)
    
    # Compute inverse frequency weights (less samples -> higher weight)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize (optional)
    return class_weights


# load previous state
def load_state(PATH, device):
    try:
        checkpoint = torch.load(PATH, map_location=device)
        return checkpoint
    except Exception as e:
        print(f"The state couldn't be loaded. Exception: {e}")
        return None

# save current state
def save_state(PATH, model, optimizer, epoch, history, scheduler=None):
    try:
        torch.save({
            'current_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'scheduler': scheduler.state_dict() if scheduler else None
        }, PATH)
    except Exception as e:
        print(f"State couldn't be saved. Exception: {e}")


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    checkpoint_path,
    checkpoint_path_best,
    epochs=10,
    device="cuda",
    n_processes=1,
    i_process=1,
    training_type="cnn",
    debug=False,
    current_epoch=1,
    current_history=None
):
    best_val_loss = float("inf")
    best_weights = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    } if not current_history else current_history
    
    
    for epoch in range(current_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
        loop.set_description(f"Epoch [{epoch}/{epochs}] - Proc: {i_process}/{n_processes}")
        
        for i, (xb, yb) in enumerate(loop, start=1):
            xb = xb.to(device)
            yb = yb.to(device).squeeze(1).long()
            
            optimizer.zero_grad() # zero the parameter gradients
            if training_type == "cnn":
                logits = model(xb)
            else:
                logits, _, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
    
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step() # update weights
    
            running_loss += loss.item() * xb.size(0)
    
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            loop.set_postfix(loss=(running_loss/total), acc=f"{(correct/total):.4f}")
            
            if debug:
                print(f"Input shape: {xb.shape}")
                print(f"Logits shape: {logits.shape}, target shape: {yb.shape}")
                print(f"Running loss: {running_loss}")
                print(f"Preds shape: {preds.shape}")
                print(f"Correct: {correct}")
                print(f"Total: {total}")
                
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        
        save_state(checkpoint_path, model, optimizer, epoch, history)
    
        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_len = len(val_dataloader)
        with torch.no_grad():
            val_loop = tqdm(val_dataloader, total=len(val_dataloader), leave=True)
            val_loop.set_description("Validation")
            
            for j, (xb, yb) in enumerate(val_loop, start=1):
                xb = xb.to(device)
                yb = yb.to(device).squeeze(1).long()

                if training_type == "cnn":
                    logits = model(xb)
                else:
                    logits, _, _ = model(xb)
                    
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
    
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()
    
                val_loop.set_postfix(loss=(val_loss/val_total), acc=f"{(val_correct/val_total):.4f}")       
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            save_state(checkpoint_path_best, model, optimizer, epoch, history)
            best_val_loss = val_loss
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    return model, history, best_val_loss