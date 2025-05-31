import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt # For optional Otsu threshold visualization
from argparse import ArgumentParser # Make sure this is imported

# --- QualiCLIP Model Loading and Preprocessing (defined globally for simplicity here) ---
# It's often better to load models inside if __name__ == '__main__' or a main function
# but for this script structure, global might be what the user intended from their snippet.
try:
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading QualiCLIP model...")
    # Added trust_repo=True as it's often needed for torch.hub.load from github
    qualiclip_model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP", trust_repo=True)
    qualiclip_model.eval().to(device)
    print("QualiCLIP model loaded successfully.")
    # QualiCLIP's Normalize parameters (standard ImageNet ones)
    normalize_transform_global = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
except Exception as e:
    print(f"Failed to load QualiCLIP model or set up device: {e}")
    qualiclip_model = None # Set to None so script can still be imported without crashing
    normalize_transform_global = None
    device = "cpu" # Default to CPU if CUDA setup failed here

def get_image_score(image_path, model, current_device, normalize_transform):
    """Computes the quality score for a single image using the QualiCLIP model."""
    if model is None or normalize_transform is None:
        print("Error: QualiCLIP model not loaded. Cannot score image.")
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        # QualiCLIP typically expects 224x224 input
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform
        ])
        img_tensor = preprocess(img).unsqueeze(0).to(current_device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(current_device.type == 'cuda')):
            score = model(img_tensor) # QualiCLIP model directly outputs the score
        return score.item()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def otsu_adaptive_thresholding_for_views(clarity_scores_map, L=256):
    """
    Calculates an adaptive threshold using Otsu's method based on input clarity scores
    and filters views deemed clear.

    Parameters:
    clarity_scores_map (dict): Dictionary mapping view identifiers to their clarity scores.
                                 Higher score means clearer.
    L (int): Number of quantization levels (bins) for the histogram of scores.
             If original scores are floats (0-1), they are scaled to an integer range 0-(L-1).

    Returns:
    tuple: (float, list)
        - otsu_threshold_original_scale (float): The calculated Otsu threshold in the original score scale.
        - clear_views_ids (list): List of view identifiers deemed clear.
    """
    if not clarity_scores_map:
        print("Warning: Clarity scores dictionary is empty. Cannot calculate Otsu threshold.")
        return 0.0, []

    scores = np.array(list(clarity_scores_map.values()))
    
    min_original_score = np.min(scores)
    max_original_score = np.max(scores)
    
    if max_original_score == min_original_score: # All scores are identical
        print(f"Warning: All images have the same score: {min_original_score:.4f}. Otsu's method may not be optimal.")
        arbitrary_mid_point = 0.5 
        if min_original_score > arbitrary_mid_point:
            print("Since all image scores are identical and relatively high, all images will be considered clear.")
            return min_original_score, list(clarity_scores_map.keys())
        else:
            print("Since all image scores are identical and relatively low (or not high), no images will be considered clear.")
            return min_original_score, []

    hist, bin_edges = np.histogram(scores, bins=L, range=(min_original_score, max_original_score))
    
    total_views = len(scores)
    probabilities = hist / float(total_views) 

    max_sigma_b_sq = 0.0
    optimal_bin_idx = 0 

    for k_bin_idx in range(L - 1): 
        omega0 = np.sum(probabilities[:k_bin_idx+1]) 
        omega1 = np.sum(probabilities[k_bin_idx+1:])

        if omega0 == 0 or omega1 == 0:
            continue

        mu0_numerator = np.sum(np.arange(k_bin_idx+1) * probabilities[:k_bin_idx+1])
        mu0 = mu0_numerator / omega0 if omega0 > 0 else 0
        
        mu1_numerator = np.sum(np.arange(k_bin_idx+1, L) * probabilities[k_bin_idx+1:])
        mu1 = mu1_numerator / omega1 if omega1 > 0 else 0
        
        sigma_b_sq = omega0 * omega1 * ((mu0 - mu1)**2)

        if sigma_b_sq > max_sigma_b_sq:
            max_sigma_b_sq = sigma_b_sq
            optimal_bin_idx = k_bin_idx
            
    otsu_threshold_original_scale = bin_edges[optimal_bin_idx + 1]

    clear_views_ids = []
    for view_id, score_val in clarity_scores_map.items():
        if score_val > otsu_threshold_original_scale: 
            clear_views_ids.append(view_id)
            
    print(f"\nOtsu Adaptive Threshold (original score scale): {otsu_threshold_original_scale:.4f}")
    print(f"Original number of views: {total_views}, Number of clear views after filtering: {len(clear_views_ids)}")

    return otsu_threshold_original_scale, clear_views_ids


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluate image quality for a folder of images using QualiCLIP and apply Otsu thresholding.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images to be evaluated.")
    parser.add_argument("--otsu_bins", type=int, default=100, help="Number of bins for Otsu's method histogram calculation (applied to scores typically in 0-1 range).")
    args = parser.parse_args()

    if qualiclip_model is None:
        print("Exiting due to QualiCLIP model loading failure.")
        exit()

    clarity_scores_map = {} 
    print(f"\nEvaluating images in folder: '{args.folder_path}'...")
    image_files = []
    for filename in os.listdir(args.folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            image_files.append(os.path.join(args.folder_path, filename))

    if not image_files:
        print("No image files found in the specified folder.")
        exit()

    for image_path in image_files:
        score = get_image_score(image_path, qualiclip_model, device, normalize_transform_global)
        if score is not None:
            clarity_scores_map[image_path] = score 
    
    if not clarity_scores_map:
        print("No images were successfully scored.")
        exit()

    print("\n--- Image scoring complete ---")

    # 1. Call Otsu's method to get the threshold and list of clear images
    otsu_thresh_val, clear_image_paths = otsu_adaptive_thresholding_for_views(
        clarity_scores_map, 
        L=args.otsu_bins
    )

    print("\n--- Clear Images (based on Otsu Threshold) ---")
    if clear_image_paths:
        for img_path in clear_image_paths:
            print(f"{os.path.basename(img_path)}: {clarity_scores_map[img_path]:.4f}")
    else:
        print("No images were classified as clear by Otsu's method.")

    # 2. (Optional) Print all image scores ranked (higher score is better)
    image_scores_list = list(clarity_scores_map.items())
    image_scores_list.sort(key=lambda x: x[1], reverse=True) 

    print("\n--- All Image Quality Scores (Ranked High to Low) ---")
    for img_path, score_val in image_scores_list:
        status = "Clear (Otsu)" if img_path in clear_image_paths else "Blurry (Otsu)"
        print(f"{os.path.basename(img_path)}: {score_val:.4f} [{status}]")

    # 3. Visualize score distribution and Otsu threshold
    if clarity_scores_map: 
        scores_for_hist = np.array(list(clarity_scores_map.values()))
        plt.figure(figsize=(12, 6))
        
        # Determine a reasonable number of bins for histogram visualization
        num_viz_bins = min(args.otsu_bins // 2, 50) if args.otsu_bins > 40 else 20
        if len(np.unique(scores_for_hist)) < num_viz_bins: # If fewer unique scores than bins
             num_viz_bins = max(1, len(np.unique(scores_for_hist)))


        plt.hist(scores_for_hist, bins=num_viz_bins, color='skyblue', edgecolor='black', alpha=0.7, label='Clarity Score Distribution')
        plt.axvline(otsu_thresh_val, color='red', linestyle='dashed', linewidth=2, label=f'Otsu Threshold: {otsu_thresh_val:.4f}')
        
        clear_scores = [s for p, s in clarity_scores_map.items() if p in clear_image_paths]
        blurry_scores = [s for p, s in clarity_scores_map.items() if p not in clear_image_paths]
        if clear_scores:
            plt.axvline(np.mean(clear_scores), color='green', linestyle='dotted', linewidth=2, label=f'Mean Score (Clear): {np.mean(clear_scores):.2f}')
        if blurry_scores:
            plt.axvline(np.mean(blurry_scores), color='orange', linestyle='dotted', linewidth=2, label=f'Mean Score (Blurry): {np.mean(blurry_scores):.2f}')

        plt.title('QualiCLIP Clarity Score Distribution and Otsu Threshold', fontsize=15)
        plt.xlabel('QualiCLIP Clarity Score (Higher is Clearer)', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        # To display the plot:
        plt.show() 
        
        # To save the plot:
        # output_plot_filename = "qualiclip_scores_otsu_visualization.png"
        # plt.savefig(output_plot_filename, dpi=300)
        # print(f"\nHistogram visualization saved to {output_plot_filename}")
        
        print("\nNote: Histogram visualization has been generated. If not displayed, ensure your environment supports a GUI.")