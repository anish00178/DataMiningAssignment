import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
mnist = fetch_openml('mnist_784', as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN model
print("Training model...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Show random test image
index = random.randint(0, len(X_test) - 1)

plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {y_pred[index]}, Actual: {y_test[index]}")
plt.axis('off')
plt.show()