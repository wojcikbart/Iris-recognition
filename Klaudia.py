from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import math

class IrisSegmentation:
    def __init__(self, image_path):
        self.image = Image.open(image_path).convert("RGB")
        self.img_array = np.array(self.image)
        self.processed_img = self.img_array.copy()
        self.gray_img = None
        self.binary_pupil = None
        self.binary_iris = None
        self.pupil_center = None
        self.pupil_radius = None
        self.iris_radius = None
        self.unwrapped_iris = None
        self.height, self.width = self.img_array.shape[:2]

    def to_grayscale(self):
        """Convert image to grayscale using luminosity method"""
        R, G, B = self.img_array[:, :, 0], self.img_array[:, :, 1], self.img_array[:, :, 2]
        self.gray_img = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
        return self.gray_img

    def compute_threshold(self, X_I=2.0, X_P=3.0):
        """Compute thresholds for iris and pupil binarization"""
        if self.gray_img is None:
            self.to_grayscale()
        
        P = np.mean(self.gray_img)
        return int(P/X_I), int(P/X_P)

    def binarize_pupil(self, threshold=None, X_P=3.0):
        """Binarize the image to isolate the pupil"""
        if self.gray_img is None:
            self.to_grayscale()
        
        if threshold is None:
            _, threshold = self.compute_threshold(X_P=X_P)
        
        self.binary_pupil = np.where(self.gray_img < threshold, 255, 0).astype(np.uint8)
        return self.binary_pupil

    def binarize_iris(self, threshold=None, X_I=2.0):
        """Binarize the image to isolate the iris"""
        if self.gray_img is None:
            self.to_grayscale()
        
        if threshold is None:
            threshold, _ = self.compute_threshold(X_I=X_I)
        
        self.binary_iris = np.where(self.gray_img < threshold, 255, 0).astype(np.uint8)
        return self.binary_iris

    def detect_pupil(self):
        """Detect pupil boundary using morphological operations and projections"""
        if self.binary_pupil is None:
            self.binarize_pupil()
        
        # Apply morphological operations to clean up the binary image
        # First close to fill holes within pupil
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(self.binary_pupil, cv2.MORPH_CLOSE, kernel)
        
        # Then open to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Find connected components (blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        # Assuming the pupil is the largest dark blob (excluding background which is label 0)
        largest_blob = 1
        largest_area = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_blob = i
        
        # Create a mask for the largest blob (pupil)
        pupil_mask = np.zeros_like(labels, dtype=np.uint8)
        pupil_mask[labels == largest_blob] = 255
        
        # Find contours in the pupil mask
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour which should be the pupil
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit circle to pupil contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            self.pupil_center = (int(x), int(y))
            self.pupil_radius = int(radius)
            
            return self.pupil_center, self.pupil_radius
        else:
            # Fallback method using projections if contour detection fails
            return self._detect_pupil_with_projections()

    def _detect_pupil_with_projections(self):
        """Fallback method to detect pupil using horizontal and vertical projections"""
        if self.binary_pupil is None:
            self.binarize_pupil()
        
        # Get horizontal and vertical projections
        h_proj = np.sum(self.binary_pupil, axis=1)
        v_proj = np.sum(self.binary_pupil, axis=0)
        
        # Find the region with highest density (pupil area)
        h_max_idx = np.argmax(h_proj)
        v_max_idx = np.argmax(v_proj)
        
        # Center of pupil should be at the maximum projections
        center_y = h_max_idx
        center_x = v_max_idx
        
        # Calculate radius by checking where the projection values drop significantly
        # This is a simple approximation
        h_threshold = 0.5 * h_proj[h_max_idx]
        v_threshold = 0.5 * v_proj[v_max_idx]
        
        h_radius = np.sum(h_proj > h_threshold) // 2
        v_radius = np.sum(v_proj > v_threshold) // 2
        
        radius = min(h_radius, v_radius)
        
        self.pupil_center = (center_x, center_y)
        self.pupil_radius = radius
        
        return self.pupil_center, self.pupil_radius

    def detect_iris(self):
        """Detect iris boundary using the pupil center and edge detection"""
        if self.pupil_center is None or self.pupil_radius is None:
            self.detect_pupil()
        
        if self.binary_iris is None:
            self.binarize_iris()
        
        # Create a mask to focus on the region around the pupil
        mask = np.zeros_like(self.gray_img)
        
        # Calculate maximum possible iris radius (typically 4-5 times pupil radius but limited by image boundaries)
        max_distance = min(
            self.pupil_center[0], 
            self.pupil_center[1], 
            self.width - self.pupil_center[0], 
            self.height - self.pupil_center[1],
            int(4.5 * self.pupil_radius)  # Max expected iris radius
        )
        
        # Apply edge detection
        edges = cv2.Canny(self.gray_img, 30, 70)
        
        # Create a circular mask for the region of interest
        y, x = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((x - self.pupil_center[0])**2 + (y - self.pupil_center[1])**2)
        
        # Focus on the ring between pupil boundary and max_distance
        mask = np.logical_and(
            dist_from_center >= self.pupil_radius + 5,  # Stay outside pupil
            dist_from_center <= max_distance
        ).astype(np.uint8) * 255
        
        # Apply mask to the edge image
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Use Hough Circle transform to find the iris boundary
        circles = cv2.HoughCircles(
            masked_edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.pupil_radius*2,
            param1=30,
            param2=20,
            minRadius=int(self.pupil_radius*1.5),
            maxRadius=int(self.pupil_radius*4)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Take the first circle that has its center close to the pupil center
                center_dist = np.sqrt((i[0] - self.pupil_center[0])**2 + (i[1] - self.pupil_center[1])**2)
                if center_dist < self.pupil_radius:  # Centers should be very close
                    self.iris_radius = i[2]
                    return self.pupil_center, self.iris_radius
        
        # Fallback: estimate iris radius based on average proportions if Hough fails
        self.iris_radius = int(self.pupil_radius * 3.2)  # Typical ratio
        return self.pupil_center, self.iris_radius

    def unwrap_iris(self, radial_resolution=80, angular_resolution=360):
        """Unwrap the iris to a rectangular representation"""
        if self.pupil_center is None or self.pupil_radius is None or self.iris_radius is None:
            self.detect_iris()
        
        if self.gray_img is None:
            self.to_grayscale()
            
        # Create the unwrapped representation
        unwrapped = np.zeros((radial_resolution, angular_resolution), dtype=np.uint8)
        
        center_x, center_y = self.pupil_center
        
        for i in range(angular_resolution):
            theta = 2.0 * math.pi * i / angular_resolution
            for j in range(radial_resolution):
                # Map j to a radius between pupil_radius and iris_radius
                r = self.pupil_radius + (self.iris_radius - self.pupil_radius) * j / radial_resolution
                
                # Convert to Cartesian coordinates
                x = int(center_x + r * math.cos(theta))
                y = int(center_y + r * math.sin(theta))
                
                # Check if the point is within the image boundaries
                if 0 <= x < self.width and 0 <= y < self.height:
                    unwrapped[j, i] = self.gray_img[y, x]
        
        self.unwrapped_iris = unwrapped
        return unwrapped

    def visualize_segmentation(self):
        """Visualize the segmentation results"""
        if self.pupil_center is None or self.pupil_radius is None or self.iris_radius is None:
            self.detect_iris()
        
        # Make a color copy of the original image for drawing
        visualization = self.img_array.copy()
        
        # Draw pupil boundary
        cv2.circle(
            visualization, 
            self.pupil_center, 
            self.pupil_radius, 
            (0, 255, 0), 
            2
        )
        
        # Draw iris boundary
        cv2.circle(
            visualization, 
            self.pupil_center, 
            self.iris_radius, 
            (255, 0, 0), 
            2
        )
        
        # Draw center point
        cv2.circle(
            visualization, 
            self.pupil_center, 
            2, 
            (0, 0, 255), 
            -1
        )
        
        return visualization

    def save_unwrapped_iris(self, filename):
        """Save the unwrapped iris to a file"""
        if self.unwrapped_iris is None:
            self.unwrap_iris()
        
        cv2.imwrite(filename, self.unwrapped_iris)

    def save_segmentation(self, filename):
        """Save the segmentation visualization to a file"""
        visualization = self.visualize_segmentation()
        cv2.imwrite(filename, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


def main():
    # Path to the input image
    image_path = "data/034/IMG_034_R_2.JPG"
    
    # Create the segmentation object
    segmentation = IrisSegmentation(image_path)
    
    # Convert to grayscale
    gray = segmentation.to_grayscale()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation.img_array)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Compute thresholds for iris and pupil

    # 2.2 i 5.3 działa zajebiście na testowym
    iris_threshold, pupil_threshold = segmentation.compute_threshold(X_I=2.2, X_P=5.3)
    print(f"Iris threshold: {iris_threshold}, Pupil threshold: {pupil_threshold}")
    
    # Binarize for pupil detection
    binary_pupil = segmentation.binarize_pupil(pupil_threshold)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binary_pupil, cmap='gray')
    plt.title("Binary Pupil")
    plt.axis('off')
    
    # Binarize for iris detection
    binary_iris = segmentation.binarize_iris(iris_threshold)
    plt.subplot(1, 2, 2)
    plt.imshow(binary_iris, cmap='gray')
    plt.title("Binary Iris")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Detect pupil
    pupil_center, pupil_radius = segmentation.detect_pupil()
    print(f"Pupil center: {pupil_center}, Pupil radius: {pupil_radius}")
    
    # Detect iris
    _, iris_radius = segmentation.detect_iris()
    print(f"Iris radius: {iris_radius}")
    
    # Visualize segmentation
    segmented = segmentation.visualize_segmentation()
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented)
    plt.title("Segmented Iris")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Unwrap iris
    unwrapped = segmentation.unwrap_iris()
    plt.figure(figsize=(12, 4))
    plt.imshow(unwrapped, cmap='gray')
    plt.title("Unwrapped Iris")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save results
    segmentation.save_unwrapped_iris("unwrapped_iris.png")
    segmentation.save_segmentation("segmented_iris.png")

if __name__ == "__main__":
    main()