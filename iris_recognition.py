from PIL import Image
from networkx import density
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
class ImageProcessor:

    def __init__(self, image_path):
        self.image = Image.open(image_path).convert("RGBA")
        self.pixels = np.array(self.image)
        self.processed_pixels = self.pixels.copy()


    def get_RGBA(self, processed=False):
        if processed:
            if self.processed_pixels.shape[-1] == 4:
                return self.processed_pixels[:, :, 0], self.processed_pixels[:, :, 1], self.processed_pixels[:, :, 2], self.processed_pixels[:, :, 3]
            else:
                return self.processed_pixels[:, :, 0], self.processed_pixels[:, :, 1], self.processed_pixels[:, :, 2]
                
        if self.pixels.shape[-1] == 4:
            return self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2], self.pixels[:, :, 3]
        return self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2], None 


    def grayscale(self, method="luminosity", processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        methods = {
            "average": ((R.astype(np.uint16) + G.astype(np.uint16) + B.astype(np.uint16)) // 3).astype(np.uint8),
            "luminosity": (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8),
            "luminance": (0.2126 * R + 0.7152 * G + 0.0722 * B).astype(np.uint8),
            "desaturation": ((np.maximum(R, G, B).astype(np.uint16) + np.minimum(R, G, B).astype(np.uint16)) // 2).astype(np.uint8),
            "decomposition-max": np.maximum(R, G, B).astype(np.uint8),
            "decomposition-min": np.minimum(R, G, B).astype(np.uint8),
        }

        if method not in methods:
            raise ValueError("Unknown grayscale method!")

        gray = methods[method]
        pixels = np.stack([gray, gray, gray, A], axis=-1) if A is not None else np.stack([gray] * 3, axis=-1)

        # self.show(pixels)
        self.processed_pixels = pixels


    def binarize(self, threshold, method="luminosity", processed=True):
        self.grayscale(method)
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        R_new = np.where(R > threshold, 255, 0).astype(np.uint8)
        G_new = np.where(G > threshold, 255, 0).astype(np.uint8)
        B_new = np.where(B > threshold, 255, 0).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)

    def compute_binarization_treshold(self, X_I=2.0, X_P=3.0, processed=True):
        self.grayscale(method="luminosity")
        gray, _, _, _ = self.get_RGBA(processed=True)
        P = np.mean(gray)
        return int(P/X_I), int(P/X_P)
    
    def _create_kernel(self, shape='rect', size=3):
        if shape == 'rect':
            return np.ones((size, size), dtype=np.uint8)
        elif shape == 'circle':
            kernel = np.zeros((size, size), dtype=np.uint8)
            radius = size // 2
            center = radius
            y, x = np.ogrid[-center:size-center, -center:size-center]
            mask = x*x + y*y <= radius*radius
            kernel[mask] = 1
            return kernel
        else:
            raise ValueError("Unknown kernel shape! Use 'rect' or 'circle'")

    def _erode(self, image, kernel):
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        result = np.zeros_like(image)
        
        for i in range(pad_h, h - pad_h):
            for j in range(pad_w, w - pad_w):
                # nieghbourhood extraction
                neighborhood = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
                
                # Apply kernel mask
                masked = neighborhood[kernel == 1]
                
                # Erosion = pixel is 1 only if all masked pixels are 1
                result[i, j] = 255 if np.all(masked == 255) else 0
                
        return result

    def _dilate(self, image, kernel):
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        result = np.zeros_like(image)
        
        for i in range(pad_h, h - pad_h):
            for j in range(pad_w, w - pad_w):
                # nieghbourhood extraction
                neighborhood = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
                
                # Apply kernel mask 
                masked = neighborhood[kernel == 1]
                
                # Dilation = pixel is 1 if any masked pixel is 1
                result[i, j] = 255 if np.any(masked == 255) else 0
                
        return result

    def erode(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        for _ in range(iterations):
            R = self._erode(R, kernel)
            G = self._erode(G, kernel)
            B = self._erode(B, kernel)
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self

    def dilate(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        for _ in range(iterations):
            R = self._dilate(R, kernel)
            G = self._dilate(G, kernel)
            B = self._dilate(B, kernel)
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self

    def morph_open(self, kernel_shape='rect', kernel_size=3, iterations=1):
        return self.erode(kernel_shape, kernel_size, iterations).dilate(kernel_shape, kernel_size, iterations)

    def morph_close(self, kernel_shape='rect', kernel_size=3, iterations=1):
        return self.dilate(kernel_shape, kernel_size, iterations).erode(kernel_shape, kernel_size, iterations)
    
    def erode2(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        for _ in range(iterations):
            R = ndimage.binary_erosion(R, structure=kernel, iterations=1).astype(np.uint8) * 255
            G = ndimage.binary_erosion(G, structure=kernel, iterations=1).astype(np.uint8) * 255
            B = ndimage.binary_erosion(B, structure=kernel, iterations=1).astype(np.uint8) * 255
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self

    def dilate2(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        for _ in range(iterations):
            R = ndimage.binary_dilation(R, structure=kernel, iterations=1).astype(np.uint8) * 255
            G = ndimage.binary_dilation(G, structure=kernel, iterations=1).astype(np.uint8) * 255
            B = ndimage.binary_dilation(B, structure=kernel, iterations=1).astype(np.uint8) * 255
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self
    
    def morph_open2(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        R = ndimage.binary_opening(R, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        G = ndimage.binary_opening(G, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        B = ndimage.binary_opening(B, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self

    def morph_close2(self, kernel_shape='rect', kernel_size=3, iterations=1):
        kernel = self._create_kernel(kernel_shape, kernel_size)
        R, G, B, A = self.get_RGBA(processed=True)
        
        R = ndimage.binary_closing(R, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        G = ndimage.binary_closing(G, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        B = ndimage.binary_closing(B, structure=kernel, iterations=iterations).astype(np.uint8) * 255
        
        if A is not None:
            self.processed_pixels = np.stack([R, G, B, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R, G, B], axis=-1)
        
        return self
    
    def reverse_pixels(self, processed=True):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        R_new = np.where(R == 0, 255, 0).astype(np.uint8)
        G_new = np.where(G == 0, 255, 0).astype(np.uint8)
        B_new = np.where(B == 0, 255, 0).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)

   
    def show(self, pixels=None):
        if pixels is None:
            pixels = self.pixels
        img = Image.fromarray(pixels)
        img.show()

    def show_processed(self):
        img = Image.fromarray(self.processed_pixels)
        img.show()

    def show_processed2(self, title="Processed Image"):
        plt.imshow(self.processed_pixels)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def save(self, output_path):
        img = Image.fromarray(self.processed_pixels)
        img.save(output_path)

    def reset(self):
        self.processed_pixels = self.pixels.copy()        

if __name__ == "__main__":
    processor = ImageProcessor("test_data/test_eye.JPG")
    threshold_iris, threshold_pupil = processor.compute_binarization_treshold(X_I=2.2, X_P=5)
    processor.reset()
    processor.grayscale(method="luminosity")
    processor.binarize(threshold_iris, processed=True)
    processor.show_processed2("Binarized Iris")
    processor.morph_open(kernel_shape='circle', kernel_size=3, iterations=1)
    processor.show_processed2("Closed Iris")
    # processor.reset()
    # processor.grayscale(method="luminosity")
    # processor.binarize(threshold_pupil, processed=True)
    # processor.show_processed()
    

    ## MORPHOLOGICAL OPERATIONS TESTS

    # test = ImageProcessor("test_data/test_open.png")
    # test.show()
    # test.morph_open(kernel_shape='circle', kernel_size=3, iterations=1)
    # test.show_processed()

    # test = ImageProcessor("test_data/test_close.png")
    # test.show()
    # test.reverse_pixels()
    # test.morph_close(kernel_shape='circle', kernel_size=3, iterations=5)
    # test.reverse_pixels()
    # test.show_processed()

