from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import cv2
import math

class IrisSegmentation:
    def __init__(self, image_path=None, image=None):
        if image_path is not None:
            self.image = Image.open(image_path).convert("RGB")
        elif image is not None:
            self.image = image
        else:
            raise ValueError("Either image_path or image must be provided")
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
        
        # Morfologiczne tutaj
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(self.binary_pupil, cv2.MORPH_CLOSE, kernel)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        largest_blob = 1
        largest_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_blob = i
        
        pupil_mask = np.zeros_like(labels, dtype=np.uint8)
        pupil_mask[labels == largest_blob] = 255
        
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            self.pupil_center = (int(x), int(y))
            self.pupil_radius = int(radius)
            
            return self.pupil_center, self.pupil_radius
        else:
            return self._detect_pupil_with_projections()

    def _detect_pupil_with_projections(self):
        """Fallback method to detect pupil using horizontal and vertical projections"""
        if self.binary_pupil is None:
            self.binarize_pupil()
        
        h_proj = np.sum(self.binary_pupil, axis=1)
        v_proj = np.sum(self.binary_pupil, axis=0)
        
        h_max_idx = np.argmax(h_proj)
        v_max_idx = np.argmax(v_proj)
        
        center_y = h_max_idx
        center_x = v_max_idx
        
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
        
        mask = np.zeros_like(self.binary_iris)
        
        max_distance = min(
            self.pupil_center[0], 
            self.pupil_center[1], 
            self.width - self.pupil_center[0], 
            self.height - self.pupil_center[1],
            int(4.5 * self.pupil_radius)
        )
        
        edges = cv2.Canny(self.binary_iris, 30, 70)
        
        y, x = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((x - self.pupil_center[0])**2 + (y - self.pupil_center[1])**2)
        
        mask = np.logical_and(
            dist_from_center >= self.pupil_radius + 5,  
            dist_from_center <= max_distance
        ).astype(np.uint8) * 255
        
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        circles = cv2.HoughCircles(
            masked_edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.pupil_radius*2,
            param1=50,  
            param2=30,  
            minRadius=int(self.pupil_radius*1.5),
            maxRadius=int(self.pupil_radius*4)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                i = i.astype(np.float64)
                
                center_dist = np.sqrt((i[0] - self.pupil_center[0])**2 + (i[1] - self.pupil_center[1])**2)
                if center_dist < self.pupil_radius:  
                    self.iris_radius = i[2]
                    return self.pupil_center, self.iris_radius
        
        self.iris_radius = int(self.pupil_radius * 2.8)  
        return self.pupil_center, self.iris_radius

    def unwrap_iris(self, radial_resolution=80, angular_resolution=360):
        """Unwrap the iris to a rectangular representation"""
        if self.pupil_center is None or self.pupil_radius is None or self.iris_radius is None:
            self.detect_iris()
        
        if self.gray_img is None:
            self.to_grayscale()
            
        
        unwrapped = np.zeros((radial_resolution, angular_resolution), dtype=np.uint8)
        
        center_x, center_y = self.pupil_center
        
        for i in range(angular_resolution):
            theta = 2.0 * math.pi * i / angular_resolution
            for j in range(radial_resolution):
                r = self.pupil_radius + (self.iris_radius - self.pupil_radius) * j / radial_resolution
                
                x = int(center_x + r * math.cos(theta))
                y = int(center_y + r * math.sin(theta))
                
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    unwrapped[j, i] = self.gray_img[y, x]
        
        self.unwrapped_iris = unwrapped
        return unwrapped

    def visualize_segmentation(self):
        """Visualize the segmentation results"""
        if self.pupil_center is None or self.pupil_radius is None or self.iris_radius is None:
            self.detect_iris()
        
        visualization = self.img_array.copy()
        
        cv2.circle(
            visualization, 
            self.pupil_center, 
            self.pupil_radius, 
            (0, 255, 0), 
            2
        )
        
        cv2.circle(
            visualization, 
            self.pupil_center, 
            int(self.iris_radius), 
            (255, 0, 0), 
            2
        )
        
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
    
    def generate_iris_code(self, radial_bands=8, frequency=0.45, angular_segments=256):
        """Generate binary iris code with proper Gabor response capture"""
        if self.unwrapped_iris is None:
            self.unwrap_iris()

        sigma = 2.0
        band_height = max(4, self.unwrapped_iris.shape[0] // radial_bands) 
        code = np.zeros((radial_bands, angular_segments * 2), dtype=np.uint8)

        t = np.linspace(-np.pi, np.pi, self.unwrapped_iris.shape[1])
        gaussian_window = np.exp(-t**2 / (2 * sigma**2))
        gabor_real = gaussian_window * np.cos(2 * np.pi * frequency * t)
        gabor_imag = gaussian_window * np.sin(2 * np.pi * frequency * t)

        gabor_real = (gabor_real - np.mean(gabor_real)) / np.linalg.norm(gabor_real)
        gabor_imag = (gabor_imag - np.mean(gabor_imag)) / np.linalg.norm(gabor_imag)

        for band in range(radial_bands):
            start = max(0, band * band_height - band_height//4)
            end = min(self.unwrapped_iris.shape[0], 
                    (band + 1) * band_height + band_height//4)
            strip = self.unwrapped_iris[start:end, :]

            smoothed = gaussian_filter(strip, sigma=0.5)
            signal = np.mean(smoothed, axis=0)
            
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            filtered_real = np.convolve(signal, gabor_real, mode='same')
            filtered_imag = np.convolve(signal, gabor_imag, mode='same')

            real_thresh = np.mean(filtered_real)
            imag_thresh = np.mean(filtered_imag)

            segment_length = len(signal) // angular_segments
            for i in range(angular_segments):
                s = i * segment_length
                e = (i + 1) * segment_length if i < angular_segments - 1 else len(signal)
                
                # Encode based on median threshold
                code[band, 2*i] = int(np.mean(filtered_real[s:e]) > real_thresh)
                code[band, 2*i+1] = int(np.mean(filtered_imag[s:e]) > imag_thresh)

        self.iris_code = code
        return code
    
    def visualize_iris_code(self):
        """Visualize the iris code"""
        if not hasattr(self, 'iris_code'):
            self.generate_iris_code()

        plt.figure(figsize=(10, 2.5))
        plt.imshow(self.iris_code, cmap='gray', interpolation='nearest', aspect='auto')
        plt.title("Iris Code Map")
        plt.xlabel("Angular bits (real + imag)")
        plt.ylabel("Radial bands")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def hamming_distance(self, other_code):
        """Calculate Hamming distance between two iris codes"""
        if self.iris_code is None:
            self.generate_iris_code()
        
        if other_code is None:
            raise ValueError("Other code must be provided for Hamming distance calculation.")
        
        if self.iris_code.shape != other_code.shape:
            raise ValueError("Iris codes must be of the same size for Hamming distance calculation.")
        
        return np.sum(self.iris_code != other_code) / np.prod(self.iris_code.shape)
    
class IrisSegmenter:
    def __init__(self):
        pass

    def prep(self, segmentation, X_I=2.2, X_P=5.3):
        segmentation.to_grayscale()
        iris_threshold, pupil_threshold = segmentation.compute_threshold(X_I=X_I, X_P=X_P)
        segmentation.binarize_pupil(pupil_threshold)
        segmentation.binarize_iris(iris_threshold)
        segmentation.detect_pupil()
        segmentation.detect_iris()
        segmentation.unwrap_iris()
        segmented = segmentation.visualize_segmentation()
        plt.figure(figsize=(10, 10))
        plt.imshow(segmented)
        plt.title("Segmented Iris")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return segmentation
    
    def compare_iris_codes(self, segmentation, iris_code1, iris_code2):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(iris_code1, cmap='gray', interpolation='nearest', aspect='auto')
        plt.title("Iris Code 1")
        plt.xlabel("Angular bits (real + imag)")
        plt.ylabel("Radial bands")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(2, 1, 2)
        plt.imshow(iris_code2, cmap='gray', interpolation='nearest', aspect='auto')
        plt.title("Iris Code 2")
        plt.xlabel("Angular bits (real + imag)")
        plt.ylabel("Radial bands")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        diff = np.abs(iris_code1 - iris_code2)
        cmap = mcolors.ListedColormap(['white', 'salmon'])
        plt.figure(figsize=(10, 5))
        plt.imshow(diff, cmap=cmap, interpolation='nearest', aspect='auto')
        plt.title("Iris Code Differences")
        plt.xlabel("Angular bits (real + imag)")
        plt.ylabel("Radial bands")
        plt.xticks([])  
        plt.yticks([])  
        plt.grid(False) 
        plt.tight_layout()
        plt.show()

        if iris_code1.shape != iris_code2.shape:
            raise ValueError("Iris codes must be of the same size for Hamming distance calculation.")
        dist = segmentation.hamming_distance(iris_code2)
        print(f"Hamming distance: {dist}")
        if dist < 0.2:
            print("Iris codes are similar.")
        else:
            print("Iris codes are different.")


def main():
    path1 = "data/041/IMG_041_L_3.JPG"
    path2 = "data/041/IMG_041_R_6.JPG"

    segmenter = IrisSegmenter()
    
    # segmentation = segmenter.prep(IrisSegmentation(path1), X_I=2.3, X_P=4.3)
    # iris_code = segmentation.generate_iris_code(angular_segments=128)

    # segmentation2 = segmenter.prep(IrisSegmentation(path2), X_I=2.2, X_P=4.2)
    # iris_code2 = segmentation2.generate_iris_code(angular_segments=128)

    # # visual iris code differences
    # segmenter.compare_iris_codes(segmentation, iris_code, iris_code2)

    ## teraz z innym okiem
    path3 = "test_eye.JPG"

    segmentation3 = segmenter.prep(IrisSegmentation(path3), X_I=2.2, X_P=5.3)
    segmentation3.generate_iris_code()
    segmentation3.visualize_iris_code()
    # segmenter.compare_iris_codes(segmentation, iris_code, iris_code3)

if __name__ == "__main__":
    main()