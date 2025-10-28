import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
import io


def generate_cam(model, image_tensor, target_class, method='gradcam'):
    """
    Generate Class Activation Map with multiple visualization methods

    Args:
        model: Trained PyTorch model
        image_tensor: Input image tensor
        target_class: Target class index
        method: CAM method ('gradcam', 'gradcam++', 'scorecam', 'ablationcam')

    Returns:
        numpy array: CAM visualization overlaid on original image
    """

    target_layers = [model.features[-1]]

    # Select CAM method
    if method == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif method == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    elif method == 'ablationcam':
        cam = AblationCAM(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    # Convert image tensor to numpy
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    # Create visualization
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    return cam_image


def generate_multi_cam_comparison(model, image_tensor, target_class):
    """
    Generate comparison of multiple CAM methods side by side

    Returns:
        PIL Image: Comparison grid
    """

    methods = ['gradcam', 'gradcam++', 'scorecam']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Generate CAMs with different methods
    for idx, method in enumerate(methods):
        try:
            cam_image = generate_cam(model, image_tensor, target_class, method)
            axes[idx + 1].imshow(cam_image)
            axes[idx + 1].set_title(method.upper(),
                                    fontsize=14, fontweight='bold')
            axes[idx + 1].axis('off')
        except Exception as e:
            axes[idx + 1].text(0.5, 0.5, f'Error: {str(e)}',
                               ha='center', va='center', transform=axes[idx + 1].transAxes)
            axes[idx + 1].axis('off')

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def generate_heatmap_only(model, image_tensor, target_class):
    """
    Generate just the heatmap without overlay

    Returns:
        numpy array: Heatmap
    """

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    # Apply colormap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def generate_segmented_regions(model, image_tensor, target_class, threshold=0.5):
    """
    Generate binary mask of highly activated regions

    Args:
        threshold: Activation threshold (0-1)

    Returns:
        tuple: (mask, annotated_image)
    """

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    # Create binary mask
    mask = (grayscale_cam > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw on original image
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    annotated = image.copy()
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    # Add text
    num_regions = len(contours)
    cv2.putText(annotated, f'Affected Regions: {num_regions}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return mask, annotated


def generate_progressive_cams(model, image_tensor, target_class, num_steps=4):
    """
    Generate progressive CAM visualizations at different layers
    Useful for understanding which layers capture what features

    Returns:
        list: List of CAM images from different layers
    """

    # Get feature layers
    feature_layers = [
        model.features[i] for i in range(len(model.features))
        if isinstance(model.features[i], torch.nn.Conv2d)
    ]

    # Select evenly spaced layers
    step = max(1, len(feature_layers) // num_steps)
    selected_layers = feature_layers[-num_steps * step::step][-num_steps:]

    cam_images = []

    for layer in selected_layers:
        try:
            cam = GradCAM(model=model, target_layers=[layer])
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

            image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1)

            cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
            cam_images.append(cam_image)
        except:
            continue

    return cam_images


def analyze_attention_statistics(model, image_tensor, target_class):
    """
    Compute statistical analysis of attention distribution

    Returns:
        dict: Statistics about attention distribution
    """

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    stats = {
        'mean_activation': float(np.mean(grayscale_cam)),
        'max_activation': float(np.max(grayscale_cam)),
        'min_activation': float(np.min(grayscale_cam)),
        'std_activation': float(np.std(grayscale_cam)),
        'high_attention_percentage': float(np.sum(grayscale_cam > 0.7) / grayscale_cam.size * 100),
        'focused_attention': float(np.sum(grayscale_cam > 0.5) / grayscale_cam.size * 100),
        'attention_concentration': float(np.max(grayscale_cam) - np.mean(grayscale_cam))
    }

    # Attention quality assessment
    if stats['attention_concentration'] > 0.4:
        stats['attention_quality'] = 'Highly Focused'
    elif stats['attention_concentration'] > 0.2:
        stats['attention_quality'] = 'Moderately Focused'
    else:
        stats['attention_quality'] = 'Diffuse'

    return stats


def create_explainability_report(model, image_tensor, target_class, image_path):
    """
    Create comprehensive explainability report with multiple visualizations

    Returns:
        dict: Complete report with visualizations and statistics
    """

    report = {}

    # 1. Standard GradCAM
    report['gradcam'] = generate_cam(model, image_tensor, target_class)

    # 2. Heatmap only
    report['heatmap'] = generate_heatmap_only(
        model, image_tensor, target_class)

    # 3. Segmented regions
    report['mask'], report['annotated'] = generate_segmented_regions(
        model, image_tensor, target_class)

    # 4. Statistics
    report['statistics'] = analyze_attention_statistics(
        model, image_tensor, target_class)

    # 5. Multi-method comparison (optional, can be expensive)
    # report['comparison'] = generate_multi_cam_comparison(model, image_tensor, target_class)

    return report
