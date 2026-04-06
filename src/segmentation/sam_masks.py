import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

checkpoint_path="../../models/sam_vit_b_01ec64.pth"

def generate_mask(image_path, output_path, checkpoint_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM model
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    if torch.cuda.is_available():
        sam.to(device="cuda")
        print("Using CUDA")
    else:
        print("Using CPU")


    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    if len(masks) == 0:
        raise ValueError("No masks generated")

    # Pick largest mask (simple heuristic)
    largest_mask = max(masks, key=lambda x: x["area"])["segmentation"]

    # Convert to image
    mask_img = (largest_mask * 255).astype(np.uint8)

    # Save mask
    cv2.imwrite(output_path, mask_img)

    print(f"Mask saved to {output_path}")
    print(cv2.imwrite(output_path, mask_img))


if __name__ == "__main__":


    generate_mask(
        "D:/Concordia/ECE/Winter 2026/COEN 691 O/Project/COEN691_vision_project/data/CO3D subsets/pizza2/images/frame000001.jpg",
        "../../data/masked_images/test_mask.png",
        checkpoint_path
    )
