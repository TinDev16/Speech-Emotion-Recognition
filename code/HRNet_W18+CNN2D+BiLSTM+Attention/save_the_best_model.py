import torch
# =========================
    # BEST MODEL
    # =========================
    if val_loss < best_val:
        best_val = val_loss
        counter = 0

        torch.save(model.state_dict(), BEST_PATH)
        print("✅ Best model saved")

    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping")
            break