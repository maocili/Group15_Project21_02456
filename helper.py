import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def download_model(url, dest_folder="models"):
    filename = url.split('/')[-1]
    full_path = os.path.join(dest_folder, filename)
    if not os.path.exists(full_path):
        print(f"File not found. Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download finished!")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"File '{filename}' already exists. Skipping download.")
    return full_path


def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def get_error_info(pred, gt):
    viz_map = np.zeros_like(gt, dtype=np.uint8)

    viz_map[(gt == 1) & (pred == 0)] = 1

    viz_map[(gt == 0) & (pred == 1)] = 2

    error_rate = np.mean(gt != pred)

    return viz_map, error_rate


def visualize_comparison_validation(loader_A, loader_B, model_A, model_B, device, save_path="dual_loader_comparison.png", samples_per_loader=3):
    model_A.eval()
    model_B.eval()

    try:
        # Get one batch from each loader
        batch_A = next(iter(loader_A))
        batch_B = next(iter(loader_B))
    except StopIteration:
        print("One of the loaders is empty!")
        return

    imgs_A, masks_A = batch_A
    imgs_B, masks_B = batch_B

    imgs_A = imgs_A[:samples_per_loader].to(device)
    masks_A = masks_A[:samples_per_loader].to(device)
    imgs_B = imgs_B[:samples_per_loader].to(device)
    masks_B = masks_B[:samples_per_loader].to(device)

    with torch.no_grad():
        print(imgs_A.shape, imgs_B.shape)
        out_A = model_A(imgs_A)
        preds_A = torch.argmax(out_A, dim=1)

        out_B = model_B(imgs_B)
        preds_B = torch.argmax(out_B, dim=1)

    imgs_A_np = imgs_A.cpu().numpy()
    imgs_B_np = imgs_B.cpu().numpy()
    masks_np = masks_A.cpu().numpy()
    preds_A_np = preds_A.cpu().numpy()
    preds_B_np = preds_B.cpu().numpy()

    batch_size = len(imgs_A)
    print(f"Visualizing combined batch size: {batch_size} ({len(imgs_A)} from A, {len(imgs_B)} from B)")

    fig, axes = plt.subplots(batch_size, 6, figsize=(24, batch_size * 4.5))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap_err = mcolors.ListedColormap(['white', 'red', 'blue'])
    norm_err = mcolors.Normalize(vmin=0, vmax=2)

    for i in range(batch_size):
        img_A_show = np.squeeze(imgs_A_np[i])
        img_B_show = np.squeeze(imgs_B_np[i])
        gt_show = np.squeeze(masks_np[i])
        pred_A = preds_A_np[i]
        pred_B = preds_B_np[i]

        iou_A = calculate_iou(pred_A, gt_show)
        iou_B = calculate_iou(pred_B, gt_show)
        err_map_A, err_rate_A = get_error_info(pred_A, gt_show)
        err_map_B, err_rate_B = get_error_info(pred_B, gt_show)

        axes[i, 0].imshow(img_A_show, cmap='gray')
        axes[i, 0].set_title(f"Original A", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # axes[i, 1].imshow(img_B_show, cmap='gray')
        # axes[i, 1].set_title(f"Original B", fontsize=10, fontweight='bold')
        # axes[i, 1].axis('off')

        axes[i, 1].imshow(gt_show, cmap='gray')
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_A, cmap='gray')
        axes[i, 2].set_title(f"Model A\nIoU: {iou_A:.4f}", fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(err_map_A, cmap=cmap_err, norm=norm_err, interpolation='nearest')
        axes[i, 3].set_title(f"Model A Error\nRate: {err_rate_A:.4f} \n Red=FN, Blue = FP", fontsize=10)
        axes[i, 3].axis('off')

        axes[i, 4].imshow(pred_B, cmap='gray')
        axes[i, 4].set_title(f"Model B\nIoU: {iou_B:.4f}", fontsize=10)
        axes[i, 4].axis('off')

        axes[i, 5].imshow(err_map_B, cmap=cmap_err, norm=norm_err, interpolation='nearest')
        axes[i, 5].set_title(f"Model B Error\nRate: {err_rate_B:.4f} \n Red=FN, Blue = FP", fontsize=10)
        axes[i, 5].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()


def visualize_single_model_3color(loader, model, device, save_path="single_model_result.png"):
    model.eval()

    # --- A. Get Batch ---
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Loader is empty!")
        return

    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)

    # --- B. Inference (Single Model) ---
    with torch.no_grad():
        output = model(images)
        # Assuming output shape is [Batch, Classes, H, W], we take argmax
        preds = torch.argmax(output, dim=1)

    # --- C. Convert to Numpy ---
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    batch_size = len(images_np)
    print(f"Visualizing batch of size: {batch_size}")

    # --- D. Plot Configuration ---
    # Changed from 6 columns to 4 columns
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, batch_size * 4.5))

    # Handle the case where batch_size is 1 (axes is not a 2D array by default)
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)

    # Define 3-Color Colormap: [0:White (Correct), 1:Red (Miss), 2:Blue (Extra)]
    cmap_err = mcolors.ListedColormap(['white', 'red', 'blue'])
    norm_err = mcolors.Normalize(vmin=0, vmax=2)

    for i in range(batch_size):
        img_show = np.squeeze(images_np[i])
        gt_show = np.squeeze(masks_np[i])
        pred_show = preds_np[i]

        # Calculate metrics (Assumes calculate_iou and get_error_info exist)
        iou = calculate_iou(pred_show, gt_show)
        err_map, err_rate = get_error_info(pred_show, gt_show)

        # 1. Original Image
        axes[i, 0].imshow(img_show, cmap='gray')
        axes[i, 0].set_title("(Original)", fontsize=10)
        axes[i, 0].axis('off')

        # 2. Ground Truth
        axes[i, 1].imshow(gt_show, cmap='gray')
        axes[i, 1].set_title("(Ground Truth)", fontsize=10)
        axes[i, 1].axis('off')

        # 3. Prediction
        axes[i, 2].imshow(pred_show, cmap='gray')
        axes[i, 2].set_title(f"Prediction\nIoU: {iou:.4f}", fontsize=10)
        axes[i, 2].axis('off')

        # 4. Error Map (3-Color)
        axes[i, 3].imshow(err_map, cmap=cmap_err, norm=norm_err, interpolation='nearest')
        axes[i, 3].set_title(f"Error Map\nRate: {err_rate:.4f}\n(Red=Miss, Blue=Extra)", fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()
