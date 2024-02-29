import cv2
import numpy as np
import argparse
from PIL import Image

def color_enhancement(I_cropped, sigma=125):
    # Convert input image to numpy array
    I_cropped = np.array(I_cropped)

    # Gaussian Blur
    g = cv2.GaussianBlur(I_cropped, (0, 0), sigma)

    # Combining Images
    alpha = 4
    beta = -4
    f1 = cv2.addWeighted(I_cropped, alpha, g, beta, 0.5)

    # Saturate
    f1 = np.clip(f1, 0, 255).astype(np.uint8)

    # Convert numpy array to PIL image
    f1 = Image.fromarray(f1) 

    return f1

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Perform color enhancement on an input image.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output image file path')
    parser.add_argument('-s', '--sigma', type=int, default=125, help='Sigma value for Gaussian blur')
    args = parser.parse_args()

    # Read input image
    input_image = cv2.imread(args.input)

    # Perform color enhancement
    enhanced_image = color_enhancement(input_image, args.sigma)

    # Write the enhanced image to output file
    cv2.imwrite(args.output, enhanced_image)

if __name__ == "__main__":
    main()
