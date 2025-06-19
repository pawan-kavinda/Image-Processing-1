import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import os

class ImageProcessor:
    
    def __init__(self, image_path):
        """Initialize image file."""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image_path = image_path
        
    def display_images(self, images, titles, figsize=(15, 10)):
        """Display multiple images in a grid."""
        n = len(images)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=figsize)
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 1. Intensity Level Reduction
    def reduce_intensity_levels(self, num_levels):
       if num_levels <= 0 or (num_levels & (num_levels - 1)) != 0:
            raise ValueError("Number of levels should be positive power of 2")

        # Calculate the scaling factor
        max_val = 255
        scale_factor = max_val / (num_levels - 1)
        
        # Quantize the image
        quantized = np.floor(self.original_image / scale_factor)
        
        # Scale back to 0-255 range
        result = (quantized * scale_factor).astype(np.uint8)
        
        return result
   
     # 2. Spatial Averaging
    def spatial_average(self, kernel_size):
      
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        averaged = cv2.filter2D(self.original_image, -1, kernel)
        return averaged
    
    def demonstrate_spatial_averaging(self):       
        kernel_sizes = [3, 10, 20]
        images = [self.original_image]
        titles = ['Original']
        
        for size in kernel_sizes:
            averaged_img = self.spatial_average(size)
            images.append(averaged_img)
            titles.append(f'{size}x{size} Average')
        
        self.display_images(images, titles)
        return images, titles

def create_sample_image():
   
    # Create a sample image with various patterns
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some geometric patterns
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
    cv2.circle(img, (150, 150), 30, 128, -1)
    cv2.line(img, (0, 0), (255, 255), 64, 2)
    
    # Add some noise
    noise = np.random.randint(0, 50, (256, 256))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite("sample_image.png", img)
    return "sample_image.png"


# Demonstration and Testing
def main():
    
    # Create or use existing image
    image_path = "sample.jpg"
    if not os.path.exists(image_path):
        print("Creating sample image for demonstration...")
        image_path = create_sample_image()
    
    try:
        # Initialize processor
        processor = ImageProcessor(image_path)
        
        print("=== Image Processing Assignment 1. ===")
        print(f"Original image shape: {processor.original_image.shape}")
        
        # 1. Intensity Level Reduction
        print("\n1. Intensity Level Reduction...")
        processor.demonstrate_intensity_reduction()

         # 2. Spatial Averaging
        print("\n2. Spatial Averaging...")
        processor.demonstrate_spatial_averaging()
        
     
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()