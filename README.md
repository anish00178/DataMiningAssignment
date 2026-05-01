📊 Data Mining Assignments using Machine Learning

A collection of beginner-friendly machine learning assignments implemented in Python using the scikit-learn library.

These assignments demonstrate how different machine learning algorithms can be used for classification problems.



🚀 Assignments Included

| Assignment                 | Dataset      | Algorithm                 | Accuracy |
| -------------------------- | ------------ | ------------------------- | -------- |
| Digit Recognition          | MNIST        | Logistic Regression       | 95–100%   |
| Digit Recognition          | MNIST        | Gaussian Naive Bayes      | 85–90%   |
| Iris Flower Classification | Iris Dataset | K-Nearest Neighbors (KNN) | 95–100%  |


1️⃣ Digit Recognition using Logistic Regression

Dataset

* MNIST handwritten digit dataset
* Contains 70,000 grayscale images of digits from 0–9
* Each image size is 28 × 28 pixels

Algorithm

Logistic Regression

Description

This program loads the MNIST dataset from OpenML, normalizes the pixel values, and splits the data into training and testing sets. A Logistic Regression model is then trained to recognize handwritten digits.

Features

* Data normalization
* Train-test split
* Model training and prediction
* Accuracy calculation
* Random digit visualization using Matplotlib


2️⃣ Digit Recognition using Gaussian Naive Bayes

Dataset

* MNIST handwritten digit dataset

Algorithm

* Gaussian Naive Bayes

Description

This assignment uses the Gaussian Naive Bayes algorithm to classify handwritten digits. The model assumes that all pixel values follow a Gaussian distribution and calculates the probability of each digit.

Features

* Simple and fast model
* Probability-based classification
* Digit prediction and accuracy evaluation
* Visualization of a predicted digit



3️⃣ Iris Flower Classification using K-Nearest Neighbors

Dataset

* Iris flower dataset
* Contains 150 flower samples
* Three flower classes:

  * Setosa
  * Versicolor
  * Virginica

Algorithm

* K-Nearest Neighbors (KNN)

Description

This program predicts the species of an iris flower using sepal and petal measurements. The KNN algorithm checks the nearest neighboring flowers and predicts the class based on majority voting.

Features

* Uses flower measurements as features
* KNN classification with k = 3
* Accuracy evaluation
* Scatter plot visualization of the dataset

4. Cancer Data Classification
Dataset: Breast Cancer Dataset (from scikit-learn)
Algorithm: Logistic Regression / Classification Model
Description: This program classifies tumors as malignant or benign using machine learning techniques based on various medical features.

What has been done:

Loaded the breast cancer dataset
Preprocessed and normalized the data
Split the dataset into training and testing sets
Trained the model using classification algorithm
Evaluated performance using accuracy score

🛠 Technologies Used

* Python
* NumPy
* Matplotlib
* scikit-learn


▶️ Installation

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```



▶️ Run the Programs

```bash
python logistic_regression_mnist.py
python naive_bayes_mnist.py
python knn_iris.py
```


📁 Project Structure

```text
Data_miningAssignments/
│
├── Cancer
├── DigitRecognition
├── iris
├── naiveBayes
└── README.md
```


📚 Dataset Sources

* MNIST dataset from OpenML
* Iris dataset from scikit-learn
* kaggle Dataset


👨‍💻 Author

Anish Akhtar
