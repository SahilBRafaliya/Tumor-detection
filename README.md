# 🧠 Tumor Detection using VGG16 + Random Forest

## 📌 Project Overview
This project focuses on **tumor detection using deep learning and machine learning techniques**. The dataset used for this project is sourced from **Kaggle**, and the model combines the power of **VGG16 (a pre-trained CNN model)** and **Random Forest** for accurate classification of tumor images.

🔗 **Dataset Source:** [Kaggle - Tumor Detection Dataset](https://www.kaggle.com/)

---
## 🎯 Features
✅ Image classification for tumor detection  
✅ Transfer learning using **VGG16** for feature extraction  
✅ Machine learning classification with **Random Forest**  
✅ Data preprocessing, augmentation, and model evaluation  

---
## 🛠️ Technologies Used
- 🐍 Python
- 🤖 TensorFlow, Keras (VGG16)
- 🌲 Scikit-learn (Random Forest)
- 📊 Pandas, NumPy
- 🖼️ OpenCV, PIL
- 📈 Matplotlib, Seaborn

---
## 🔍 Techniques Used

### 1️⃣ Data Preprocessing
🖼️ **Image Loading & Resizing:** Converted images to a uniform size (224x224) for VGG16 compatibility  
🎨 **Data Augmentation:** Applied rotation, flipping, and brightness adjustments to enhance generalization  
📂 **Train-Test Split:** Divided dataset into training and testing sets for model evaluation  

### 2️⃣ Feature Extraction with VGG16
🔄 Used **VGG16 (pre-trained on ImageNet)** to extract deep features from tumor images  
⚙️ Removed the fully connected layers to retain only convolutional feature maps  
📊 Extracted **deep feature vectors** to serve as input for the Random Forest model  

### 3️⃣ Classification with Random Forest
🌲 **Trained Random Forest classifier** on the extracted VGG16 feature vectors  
📌 Optimized hyperparameters (n_estimators, max_depth) to improve classification performance  
📉 Evaluated model using **accuracy, precision, recall, and F1-score**  

### 4️⃣ Model Evaluation & Insights
📊 **Performance Metrics:** Assessed accuracy, confusion matrix, and classification report  
📌 **Comparison of VGG16+RandomForest vs. CNN-based approaches**  
📈 **Visualized feature importance from Random Forest**  

---
## 🚀 How to Use
1️⃣ Load the dataset from Kaggle  
2️⃣ Run the **preprocessing script** to prepare images for VGG16  
3️⃣ Execute the **VGG16 feature extraction notebook**  
4️⃣ Train the **Random Forest classifier** using the extracted features  
5️⃣ Evaluate model performance and analyze classification results  

---
## 🔮 Future Enhancements
📢 **Experiment with other CNN architectures** (ResNet, EfficientNet, MobileNet)  
📊 **Hyperparameter tuning** for both VGG16 and Random Forest models  
📡 **Deploy as a web application** using Flask or Streamlit  
🧠 **Integrate with real-world medical imaging datasets** for clinical validation  

---
## 📌 Conclusion
This project demonstrates how **transfer learning with VGG16** and **Random Forest classification** can effectively detect tumors. By leveraging deep feature extraction and machine learning, we can achieve accurate medical image classification.

---
## 📚 References
- Kaggle Dataset: [kaggle.com](https://www.kaggle.com/)
- TensorFlow/Keras: [tensorflow.org](https://www.tensorflow.org/)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org/)
- Matplotlib: [matplotlib.org](https://matplotlib.org/)
