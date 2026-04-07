import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(BASE_DIR, "../../models/sam_vit_b_01ec64.pth")


def generate_mask(image_path, output_path, mask_generator):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    if len(masks) == 0:
        raise ValueError("No masks generated")

    # Pick largest mask (simple heuristic)
    largest_mask = max(masks, key=lambda x: x["area"])["segmentation"]

    # Convert mask to 3 channels
    mask_3ch = np.stack([largest_mask] * 3, axis=-1)

    # Apply mask to original image
    masked_image = image * mask_3ch

    # Save masked image
    cv2.imwrite(output_path, masked_image)

    print(f"Masked image saved to {output_path}")

def run_sam_folder(config):
    input_folder = config["data"]["images_path"]
    output_folder = config["data"]["masked_images_path"]

    os.makedirs(output_folder, exist_ok=True)

    # Load SAM model ONCE
    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    if torch.cuda.is_available():
        sam.to(device="cuda")
        print("Using CUDA")
    else:
        print("Using CPU")

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Loop through images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing {filename}...")

            generate_mask(
                input_path,
                output_path,
                mask_generator
            )
