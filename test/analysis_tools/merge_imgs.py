import os
from PIL import Image

def weighted_image_fusion(folder_path, output_size=(320, 320), weights=None, output_filename="fused_image.jpg"):
    """
    Resize and fuse images in a folder by blending them according to specified weights.
    The fused image is saved directly in the folder.

    Args:
        folder_path (str): Path to the folder containing the images.
        output_size (tuple): Target size (width, height) to resize each image.
        weights (list): List of weights (transparency) for blending each image. If None, equal weights are used.
        output_filename (str): The name of the output fused image file.
    """
    # List all image files in the directory
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    if len(image_files) == 0:
        raise ValueError("No images found in the specified folder.")
    
    # If weights are not provided, assign equal weight to each image
    if weights is None:
        weights = [1 / len(image_files)] * len(image_files)
    elif len(weights) != len(image_files):
        raise ValueError("The number of weights must match the number of images.")
    
    # Open and resize all images, then blend them according to the weights
    base_image = None
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).resize(output_size).convert("RGBA")
        
        if base_image is None:
            # Initialize the base image as the first image
            base_image = img
        else:
            # Blend the base image with the current image using the weight
            base_image = Image.blend(base_image, img, alpha=weights[i])
    
    # Convert the image back to RGB mode before saving
    fused_image = base_image.convert("RGB")
    
    # Save the fused image
    output_path = os.path.join(folder_path, output_filename)
    fused_image.save(output_path)
    print(f"Fused image saved at {output_path}")

if __name__ == "__main__":
    folder_path = "./data/output/fuse/sample2/UNet/Sag"
    weighted_image_fusion(folder_path, output_size=(320, 320), weights=None)
