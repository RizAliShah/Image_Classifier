# 🖼️ Emotion Image Classifier (Happy vs Sad)  

This project is a **deep learning-based image classifier** built using **TensorFlow/Keras** and **MobileNetV2** as the backbone.  
It classifies facial expressions into **Happy** 😀 or **Sad** 😢 categories with high accuracy.  

---

## 🚀 Features
- Transfer Learning with **MobileNetV2** (pre-trained on ImageNet).  
- Custom head with **Dense layers + Dropout** for classification.  
- Support for **data augmentation** (flip, rotation, zoom).  
- Tracks **precision, recall, F1-score, and confusion matrix**.  
- Compatible with **Google Colab / Jupyter Notebook**.  

---

## 📂 Project Structure
Emotion-Image-Classifier/  
│── dataset/ # Your dataset (Happy, Sad)  
│── notebooks/ # Jupyter/Colab notebooks  
│── models/ # Saved models (.h5 or .keras)  
│── results/ # Training logs, plots, confusion matrix  
│── README.md # Project documentation  
│── requirements.txt # Dependencies  

---

## 🛠️ Installation

Clone the repository:
```
git clone https://github.com/your-username/emotion-image-classifier.git
cd emotion-image-classifier
```

Install dependencies:
```
pip install -r requirements.txt
```
If running on Google Colab, just install directly:
```
!pip install -q tensorflow tensorflow-gpu opencv-python matplotlib scikit-learn seaborn
```

## 📊 Dataset

The model is trained on a small custom dataset with two classes:

* Happy 😀

* Sad 😢

👉 You can replace it with your own dataset.
👉 Make sure the dataset is structured like this:

dataset/  
│── train/  
│   ├── Happy/  
│   └── Sad/  
│── test/  
    ├── Happy/  
    └── Sad/  

## 🧠 Model Architecture

* Base model: MobileNetV2 (pre-trained on ImageNet, partially unfrozen for fine-tuning).

Added layers:

* Global Average Pooling

* Dense (ReLU)

* Dropout (0.2–0.5)

* Output layer (Softmax, 2 classes)

## 📈 Training & Evaluation

The model is trained using:

* Optimizer: Adam

* Loss: SparseCategoricalCrossentropy

* Metrics: Accuracy, Precision, Recall

* Sample Results (on test set, 49 images):

      precision    recall  f1-score   support

Happy         0.96        0.77        0.85          30  
Sad           0.72        0.95        0.82          19

accuracy                           0.84        49  
macro avg       0.84      0.86      0.84        49  
weighted avg    0.87      0.84      0.84        49  

## 📊 Visualization

* Confusion Matrix

* Accuracy & Loss Curves

* Sample Predictions

Example Confusion Matrix:

[[23   7]  
 [ 1  18]]

## 🔮 Future Improvements

* Expand dataset with more diverse samples.

* Fine-tune deeper layers of MobileNetV2.

* Experiment with ResNet50 / EfficientNet.

* Add multi-class support (more emotions).

## 🧑‍💻 Author

Dr. Rizwan
Assistant Professor, IT Department, Islamia University of Bahawalpur
Specialization: Cybersecurity, Deep Learning, Image/Video Content Protection

## 📜 License

This project is licensed under the MIT License.
