**AI Pneumonia Detection from Chest X-Rays (VGG16)**
This project builds a deep learning model that classifies chest X-ray images as NORMAL or PNEUMONIA using transfer learning with VGG16. The goal is to simulate a simplified AI-assisted radiology tool that can identify patterns associated with pneumonia in medical imaging.
The model leverages a pre-trained convolutional neural network (CNN) to extract visual features and applies a custom classification head to perform binary classification.

**Key Features**
- Transfer learning using VGG16 (ImageNet weights)
- Image classification on medical X-ray data
-  Data augmentation to improve generalization
-  Performance evaluation with accuracy, precision, recall, and confusion matrix
-  Single-image prediction function
-  Training visualization (loss, accuracy, precision, recall)
-  What This Project Actually Does (Simple Explanation)

The model learns patterns like:
- cloudiness
- opacity
- irregular textures in lung regions
After training, it can take a new X-ray image and output:
- a prediction (NORMAL or PNEUMONIA)
- a probability score

**Model Architecture**
This project uses VGG16 as a feature extractor:
_1. Base Model (Frozen):_
- Pre-trained VGG16 CNN
- Extracts low-level and high-level image features (edges → textures → shapes)
_2. Custom Classification Head:_
- Global Average Pooling
- Dense layer (ReLU)
- Dropout (reduces overfitting)
- Output layer (Sigmoid for binary classification)

**Dataset**
Dataset used:
Chest X-Ray Images (Pneumonia) (Kaggle)
Structure:

chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
    
**Installation & Setup**
1. Clone the repository
git clone https://github.com/yourusername/pneumonia-detector.git
cd pneumonia-detector
2. Install dependencies
pip install tensorflow matplotlib numpy scikit-learn
3. Download dataset

Download from Kaggle and place it in your project directory.

**How to Run**
Option 1: Jupyter Notebook
jupyter notebook

Open:
AI_Pneumonia_Detection_VGG16.ipynb
Option 2: Google Colab

Upload the notebook and run all cells.

**Model Training Details:**
- Input size: 224 × 224
- Batch size: 32
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metrics:
- Accuracy
- Precision

In a medical context, missing a pneumonia case (false negative) is more dangerous than a false positive, so recall is especially important.

**Example Prediction**
predict_xray("test_image.jpeg")

Output:
Prediction: PNEUMONIA
Probability: 92.4%
VGG16 captures strong visual features even for medical images
Data augmentation helps prevent overfitting
Model performance improves further with fine-tuning
