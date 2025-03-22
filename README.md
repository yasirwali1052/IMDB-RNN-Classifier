# IMDB-RNN-Classifier
 A movie review sentiment analysis project using **RNN (Recurrent Neural Network)**. The model is trained on the **IMDB dataset** to classify reviews as positive or negative. The trained model is deployed on **Streamlit** for an interactive user interface. 🚀

 # IMDB Movie Review Sentiment Analysis

## 📌 Overview
The **IMDB Movie Review Sentiment Analysis** project is a deep learning-based application that classifies movie reviews as **positive** or **negative** using a **Simple RNN (Recurrent Neural Network)** model. This project leverages the **IMDB dataset**, TensorFlow's **Keras API**, and **Streamlit** for an interactive web interface.

## 🔥 Features
- Loads the **IMDB dataset** with a vocabulary size of 10,000 words.
- Preprocesses text data using **word indexing and padding**.
- Implements a **Simple RNN** model with **ReLU activation**.
- Uses **binary cross-entropy loss** and **Adam optimizer** for training.
- Includes **early stopping** to prevent overfitting.
- Saves the trained model as `simple_rnn_imdb.h5`.
- Provides a **Streamlit web app** for real-time sentiment analysis.

## 🏗️ Project Structure
```
IMDB-RNN-Classifier/
│-- simplearn.ipynb         # Jupyter Notebook for training the model
│-- prediction.ipynb        # Notebook for testing the model
│-- main.py                 # Streamlit app for user interaction
│-- simple_rnn_imdb.h5      # Trained RNN model
│-- requirements.txt        # Dependencies for the project
│-- README.md               # Project documentation
```

## 📥 Dataset
This project uses the **IMDB dataset** from `tensorflow.keras.datasets.imdb`, which contains **50,000** movie reviews labeled as **positive** (1) or **negative** (0).

- **Training Set**: 25,000 labeled reviews
- **Testing Set**: 25,000 labeled reviews
- **Vocabulary Size**: 10,000 words
- **Maximum Review Length**: 500 words (after padding)

## 🏗️ Model Architecture
The model consists of:
1. **Embedding Layer**: Converts words into dense vectors.
2. **Simple RNN Layer**: Processes sequences of words.
3. **Dense Output Layer**: Uses a **sigmoid activation** for binary classification.

### **🔧 Model Summary**
```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=500),
    SimpleRNN(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## 🏋️‍♂️ Training the Model
The model is trained using **binary cross-entropy loss** and an **Adam optimizer** with early stopping.
```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
```

## 🎯 Prediction Pipeline
1. Convert input text into numerical sequences using **word_index**.
2. Apply **padding** to ensure uniform input size.
3. Predict sentiment using the trained **RNN model**.
4. Output **positive** or **negative** sentiment along with a confidence score.

### **🔎 Example Prediction**
```python
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."
sentiment, score = predict_sentiment(example_review)
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {score}')
```

## 🌐 Streamlit Web App
The **Streamlit** web application allows users to enter a movie review and get a sentiment classification.

### **🔧 Run the Web App**
```bash
streamlit run main.py
```

### **💻 Streamlit UI Features**
- **Text input**: Users enter a movie review.
- **Button**: Click "Classify" to get sentiment prediction.
- **Output**: Displays **positive** or **negative** sentiment with a confidence score.

## 🔧 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yasirwali1052/IMDB-RNN-Classifier.git
cd IMDB-RNN-Classifier
```
### **2️⃣ Install Dependencies**
Create a virtual environment and install the required packages.
```bash
pip install -r requirements.txt
```
### **3️⃣ Run the Project**
#### **Train the Model (if needed)**
```bash
python simplearn.ipynb
```
#### **Run Predictions in Jupyter Notebook**
```bash
python prediction.ipynb
```
#### **Run Streamlit Web App**
```bash
streamlit run main.py
```

## 📊 Performance Metrics
- **Training Accuracy**: ~85%
- **Validation Accuracy**: ~83%


