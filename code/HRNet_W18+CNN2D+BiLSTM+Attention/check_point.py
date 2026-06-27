import torch 
# =========================
    # SAVE CHECKPOINT EVERY EPOCH
    # =========================
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "counter": counter
    }, CKPT_PATH)

    # optional: Save each epoch's model state_dict for later analysis or ensemble methods
    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
    )
