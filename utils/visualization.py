
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curve(history, path):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = ax1.twinx()
    ax2.plot(history["val_auc"], label="Validation ROC-AUC")
    ax2.set_ylabel("Validation ROC-AUC")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
