#feature_extractor.py

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image #load and preprocess images
from tensorflow.keras.models import Model #create a model that outputs features from intermediate layers of a neural network
import numpy as np

#Loading pretrained model :

class FeatureExtractor:
    def __init__(self):
        # Load the ResNet50 model with pretrained weights
        base_model = ResNet50(weights='imagenet')
        
        # Extract features insted of making predictions. from the 'avg_pool' layer (before the dense layers) that outputs 2048-dimensional feature vector
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def extract(self, img):
       
        img = img.resize((224, 224))  # required size for resnet50
        
        img = img.convert('RGB')  # converting/ensuring RGB
        
        x = image.img_to_array(img)  # Convert to np.array (height x width x channels)
        
        x = np.expand_dims(x, axis=0)  # Expand dimensions for batch size (1, h, w, c) Keras models expect inputs in batches.
        
        x = preprocess_input(x)  # Preprocess using ResNet50 specific preprocessing eg.scaling pixel values
        
        feature = self.model.predict(x)[0]  # Extract the feature
        
        return feature / np.linalg.norm(feature)  # Normalize the feature vector
