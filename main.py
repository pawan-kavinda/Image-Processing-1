import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import os

class ImageProcessing:
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

    def demonstrate_intensity_reduction(self):
        levels = [2, 4, 8, 16, 32, 64]
        images = [self.original_image]
        titles = ['Original (256 levels)']

        for level in levels:
            reduced_img = self.reduce_intensity_levels(level)
            images.append(reduced_img)
            titles.append(f'{level} levels')

        self.display_images(images, titles, figsize=(18, 12))
        return images, titles

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

    # 3. Image Rotation
    def rotate_image(self, angle):
        if angle == 90:
            # Use OpenCV for 90-degree rotation (more efficient)
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_180)
        else:
            # Use scipy for arbitrary angles
            rotated = rotate(self.original_image, angle, reshape=True, mode='constant', cval=0)
            rotated = np.clip(rotated, 0, 255).astype(np.uint8)

        return rotated

    def demonstrate_rotation(self):
        angles = [45, 90]
        images = [self.original_image]
        titles = ['Original']

        for angle in angles:
            rotated_img = self.rotate_image(angle)
            images.append(rotated_img)
            titles.append(f'Rotated {angle}°')

        self.display_images(images, titles)
        return images, titles

    # 4. Block-wise Resolution Reduction
    def reduce_resolution_blocks(self, block_size):
        h, w = self.original_image.shape
        result = self.original_image.copy().astype(np.float32)

        # Process non-overlapping blocks
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                # Extract block
                block = self.original_image[i:i+block_size, j:j+block_size]

                # Calculate average
                avg_value = np.mean(block)

                # Replace all pixels in block with average
                result[i:i+block_size, j:j+block_size] = avg_value

        return result.astype(np.uint8)

    def demonstrate_resolution_reduction(self):
        block_sizes = [3, 5, 7]
        images = [self.original_image]
        titles = ['Original']

        for size in block_sizes:
            reduced_img = self.reduce_resolution_blocks(size)
            images.append(reduced_img)
            titles.append(f'{size}x{size} Blocks')

        self.display_images(images, titles)
        return images, titles

    def save_results(self, output_dir="results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save original
        cv2.imwrite(f"{output_dir}/original.png", self.original_image)

        # Save intensity reduction results
        levels = [2, 4, 8, 16, 32, 64]
        for level in levels:
            reduced_img = self.reduce_intensity_levels(level)
            cv2.imwrite(f"{output_dir}/intensity_{level}_levels.png", reduced_img)

        # Save spatial averaging results
        kernel_sizes = [3, 10, 20]
        for size in kernel_sizes:
            averaged_img = self.spatial_average(size)
            cv2.imwrite(f"{output_dir}/spatial_avg_{size}x{size}.png", averaged_img)

        # Save rotation results
        angles = [45, 90]
        for angle in angles:
            rotated_img = self.rotate_image(angle)
            cv2.imwrite(f"{output_dir}/rotated_{angle}_degrees.png", rotated_img)

        # Save resolution reduction results
        block_sizes = [3, 5, 7]
        for size in block_sizes:
            reduced_img = self.reduce_resolution_blocks(size)
            cv2.imwrite(f"{output_dir}/resolution_reduced_{size}x{size}.png", reduced_img)

        print(f"All results saved to '{output_dir}' directory")

# Demonstration and Testing
def main():
   
    image_path = "sample.png"   
    try:
        # Initialize processor
        processor = ImageProcessing(image_path)

        print("=== Image Processing Assignment 1. ===")
        print(f"Original image shape: {processor.original_image.shape}")

        # 1. Intensity Level Reduction
        print("\n1. Intensity Level Reduction...")
        processor.demonstrate_intensity_reduction()

        # 2. Spatial Averaging
        print("\n2. Spatial Averaging...")
        processor.demonstrate_spatial_averaging()

        # 3. Image Rotation
        print("\n3. Image Rotation...")
        processor.demonstrate_rotation()

        # 4. Resolution Reduction
        print("\n4. Resolution Reduction...")
        processor.demonstrate_resolution_reduction()

        # Save all results
        print("\n5. Saving Results...")
        processor.save_results()

        print("\nAll operations completed successfully!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()