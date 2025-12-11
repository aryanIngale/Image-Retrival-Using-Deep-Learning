#  Image Retrieval Using Deep Learning

This project implements a **Content-Based Image Retrieval (CBIR)** system capable of finding visually similar images using deep learning. Instead of relying on text tags or manual annotations, the system analyzes the visual characteristics of images using **ResNet50** feature embeddings and ranks similarity using cosine distance. A Flask-based web interface allows users to upload a query image and instantly view the most relevant results from the dataset.

---

##  Project Information

Image retrieval is an essential task in domains where visual similarity matters—such as e-commerce, wildlife species identification, medical imaging, surveillance, and digital asset management.  
This project demonstrates how deep learning can be used to extract meaningful representations from images and build an efficient retrieval mechanism on top of them.

The system consists of three core modules:
- **Offline feature extraction**
- **Deep feature engine (ResNet50)**
- **Flask web application for retrieval results**

---

##  Vision of the Project

The vision behind the project is to build a retrieval system that:
- Understands visual content rather than text  
- Produces accurate similarity matches even across varying angles, lighting, or colors  
- Works efficiently with large datasets  
- Provides a clean and user-friendly interface  
- Is extendable to real-world applications such as product search, face recognition, object clustering, and dataset organization  

By leveraging deep learning, the system avoids the limitations of manual tagging and retrieves images purely based on learned visual features.

---

##  My Role & Contributions

I contributed primarily to the **core functional engine of this project**, which includes deep feature extraction and dataset indexing.

###  FeatureExtractor.py  
I implemented the complete feature extraction pipeline using **ResNet50 (ImageNet pretrained)**:
- Processed images (RGB conversion, resizing, preprocessing)  
- Generated robust **2048-dimensional embeddings**  
- Normalized feature vectors to improve similarity accuracy  
- Built a reusable class for backend integration  
- Ensured performance suitable for real-time retrieval  

###  Offline.py  
I developed the batch-processing script responsible for generating and storing feature vectors for the entire dataset:
- Converted dataset images into optimized NumPy feature files  
- Saved them with correct mapping to image paths  
- Ensured consistency with the server’s retrieval mechanism  

These two modules form the **core intelligence** of the project — powering similarity search and enabling fast inference during user queries.

---

##  How the System Works

1. ResNet50 extracts deep features from each dataset image  
2. These features are saved offline as `.npy` vectors  
3. When a user uploads a query image:  
   - Its features are extracted using the same pipeline  
   - Cosine similarity is computed against stored vectors  
4. The system returns the **Top-K most similar images**  
5. Flask renders the results in a clean, responsive interface  

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras (ResNet50)**
- **NumPy**
- **scikit-learn (Cosine Similarity)**
- **Pillow (Image Processing)**
- **Flask (Web Framework)**
- **HTML / CSS / Bootstrap (UI)**

---

##  Project Screenshots
**UI Interface**
<img width="1920" height="1080" alt="Screenshot (9)" src="https://github.com/user-attachments/assets/bb37b608-55b1-42da-94fd-821b1f3e9d7d" />



**Search Engine**
<img width="1920" height="1080" alt="Screenshot (10)" src="https://github.com/user-attachments/assets/8374c6b7-be7f-4839-a8df-b688dde82eb0" />

