# offline.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #tells TensorFlow to suppress non-critical warning messages

import numpy as np
from pathlib import Path   # object-oriented way to handle file system paths
from PIL import Image
from feature_extractor import FeatureExtractor

# Initialize the feature extractor
fe = FeatureExtractor()

# Path to the images you want to process
image_dir = Path("./static/img")

# Path to save the features
feature_dir = Path("./static/feature")

feature_dir.mkdir(parents=True, exist_ok=True)   #Creates the feature_dir if it doesn’t already exist

# Iterate through all images in the specified directory (JPG and PNG)
for img_path in image_dir.glob("*.[jp][pn]g"):  # Matches .jpg, .jpeg, .png ensures the loop only processes these specific file types.
    img = Image.open(img_path)  # Load the image
    features = fe.extract(img)  # Extract features

    # Save features to a .npy file
    feature_file = feature_dir / f"{img_path.stem}.npy" # Extracts the file name without the extension
    np.save(feature_file, features)
    
    # printing message that X image and its feature saved to..file location
    print(f"Processed {img_path.name} and saved features to {feature_file.name}")