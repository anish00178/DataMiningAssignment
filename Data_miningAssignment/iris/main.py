import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model using Support Vector Machine
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Step 6: Pick a random flower from test set
index = random.randint(0, len(X_test) - 1)

# Step 7: Visualize first two features
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset Visualization using SVM")
plt.show()

# Step 8: Show prediction
print("Flower Features:", X_test[index])
print("Predicted:", iris.target_names[y_pred[index]])
print("Actual:", iris.target_names[y_test[index]])