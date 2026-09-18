"""
evaluate.py
-----------
Two jobs:

1. `compute_threshold()`
   Runs the trained model on validation "good" images and records their
   reconstruction error. Since the model has never seen defects, good images
   should have LOW error and defective images should have HIGH error.
   We set the anomaly threshold as (mean + 3*std) of the good-image errors —
   basically "anything much worse than what we've seen from clean parts is
   flagged as an anomaly."

2. `evaluate_test_set()`
   Runs the model on both good + defective test images, computes an
   anomaly score per image, compares against the threshold, and reports
   Accuracy, Precision, Recall, F1, and AUC-ROC. Also saves side-by-side
   heatmap visualizations to outputs/.

Run:
    python src/evaluate.py
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from model import ConvAutoencoder
from dataset import FolderImageDataset
from torch.utils.data import DataLoader


def get_reconstruction_errors(model, dataloader, device):
    """Returns per-image mean-squared reconstruction error and the raw error maps."""
    model.eval()
    errors = []
    error_maps = []
    images_list = []
    recon_list = []
    paths_all = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            reconstructions = model(images)

            # per-pixel squared error. Instead of averaging over the whole image
            # (which dilutes small, localized defects), we take the 99th percentile
            # of error values per image -> "how bad is the worst 1% of this image?"
            # This makes small scratches/dents/holes much easier to detect.
            err_map = (images - reconstructions) ** 2
            err_map_channel_avg = err_map.mean(dim=1)  # [B, H, W]
            flat = err_map_channel_avg.view(err_map_channel_avg.size(0), -1)
            k = max(1, int(0.99 * flat.size(1)))
            per_image_error = torch.kthvalue(flat, k, dim=1).values.cpu().numpy()

            errors.extend(per_image_error.tolist())
            error_maps.extend(err_map.mean(dim=1).cpu().numpy())  # collapse channels for heatmap
            images_list.extend(images.cpu().numpy())
            recon_list.extend(reconstructions.cpu().numpy())
            paths_all.extend(paths)

    return np.array(errors), error_maps, images_list, recon_list, paths_all


def compute_threshold(model, good_val_dir, device, std_multiplier=0.5):
    dataset = FolderImageDataset(good_val_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    errors, *_ = get_reconstruction_errors(model, loader, device)

    mean_err, std_err = errors.mean(), errors.std()
    threshold = mean_err + std_multiplier * std_err
    print(f"Good-image error -> mean: {mean_err:.6f}, std: {std_err:.6f}")
    print(f"Chosen anomaly threshold: {threshold:.6f}")
    return threshold


def evaluate_test_set(model, good_test_dir, defective_test_dir, threshold, device, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    good_ds = FolderImageDataset(good_test_dir)
    def_ds = FolderImageDataset(defective_test_dir)
    good_loader = DataLoader(good_ds, batch_size=16, shuffle=False)
    def_loader = DataLoader(def_ds, batch_size=16, shuffle=False)

    good_errors, good_maps, good_imgs, good_recons, good_paths = get_reconstruction_errors(model, good_loader, device)
    def_errors, def_maps, def_imgs, def_recons, def_paths = get_reconstruction_errors(model, def_loader, device)

    all_errors = np.concatenate([good_errors, def_errors])
    all_labels = np.concatenate([np.zeros(len(good_errors)), np.ones(len(def_errors))])  # 0=good, 1=defective
    all_preds = (all_errors > threshold).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_errors)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AUC-ROC  : {auc:.3f}")

    # Save a few example heatmaps (input | reconstruction | error heatmap)
    save_heatmap_examples(good_imgs[:2], good_recons[:2], good_maps[:2], out_dir, prefix="good")
    save_heatmap_examples(def_imgs[:4], def_recons[:4], def_maps[:4], out_dir, prefix="defective")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def save_heatmap_examples(images, recons, error_maps, out_dir, prefix):
    for i, (img, recon, emap) in enumerate(zip(images, recons, error_maps)):
        img_disp = np.transpose(img, (1, 2, 0))
        recon_disp = np.transpose(recon, (1, 2, 0))

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].imshow(img_disp); axs[0].set_title("Input"); axs[0].axis("off")
        axs[1].imshow(recon_disp); axs[1].set_title("Reconstruction"); axs[1].axis("off")
        axs[2].imshow(emap, cmap="hot"); axs[2].set_title("Anomaly Heatmap"); axs[2].axis("off")

        save_path = os.path.join(out_dir, f"{prefix}_example_{i}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/autoencoder_best.pth")
    parser.add_argument("--good_val_dir", type=str, default="data/train/good")  # or a held-out good split
    parser.add_argument("--good_test_dir", type=str, default="data/test/good")
    parser.add_argument("--defective_test_dir", type=str, default="data/test/defective")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    threshold = compute_threshold(model, args.good_val_dir, device)
    evaluate_test_set(model, args.good_test_dir, args.defective_test_dir, threshold, device)
