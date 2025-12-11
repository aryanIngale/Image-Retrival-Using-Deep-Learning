import sys
# Force stdout/stderr to utf-8 so Windows won't choke on unicode logs from TF/Keras
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # Older Pythons might not have reconfigure
    pass



from flask import Flask, request, render_template
import numpy as np
from pathlib import Path
from PIL import Image
from feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the feature extractor
fe = FeatureExtractor()

# Load pre-extracted feature vectors from saved .npy files
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    
    # Load corresponding image paths (JPG or PNG)
    img_path_jpg = Path("./static/img") / (feature_path.stem + ".jpg")
    img_path_png = Path("./static/img") / (feature_path.stem + ".png")

    if img_path_jpg.exists():
        img_paths.append(img_path_jpg)
    elif img_path_png.exists():
        img_paths.append(img_path_png)

# Convert the list of features into a NumPy array
features = np.array(features)

@app.route("/about")
def about():
    return render_template('about.html')  # Create this template as needed

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded image file from form
        file = request.files['query_img']
        # Ensure the uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded image to the specified upload folder
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # Save the file
        file.save(upload_path)
        
        img = Image.open(file.stream)  # PIL image

        # Extract features from the query image
        query_features = fe.extract(img)
        
        # Calculate cosine similarity between the query and stored features
        similarities = cosine_similarity([query_features], features)
        
        # Get indices of the most similar images (sorted by similarity)
        ranked_indices = np.argsort(similarities[0])[::-1]

        # Get top 5 most similar images
        top_img_paths = [img_paths[idx] for idx in ranked_indices[:5]]
        
        # Render the result page with the top similar images
        return render_template('index.html', query_path=file.filename, result_paths=top_img_paths)

    # If GET request, show the upload form
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
