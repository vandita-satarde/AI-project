import cv2
import matplotlib.pyplot as plt

from utils import load_images
from pca import apply_pca
from ann import train_ann
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 🔹 Function for new image prediction
def predict_image(img_path, mean, model, label_map):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (100, 100)).flatten()

    img_centered = img - mean
    prediction = model.predict([img_centered])

    return label_map[prediction[0]]


# 🔹 Load data
X, y, label_map = load_images("dataset")

# 🔹 Split (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# 🔹 Apply PCA
k = 50
eigenfaces, mean, X_centered = apply_pca(X_train, k)

# 🔹 Train ANN
model = train_ann(X_centered, y_train)

# 🔹 Test
X_test_centered = X_test - mean
predictions = model.predict(X_test_centered)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# 📊 Step 6: Accuracy vs k graph
k_values = [10, 20, 30, 40, 50]
accuracies = []

for k in k_values:
    eigenfaces, mean_k, X_centered_k = apply_pca(X_train, k)
    model_k = train_ann(X_centered_k, y_train)

    X_test_centered_k = X_test - mean_k
    pred_k = model_k.predict(X_test_centered_k)

    acc = accuracy_score(y_test, pred_k)
    accuracies.append(acc)

plt.plot(k_values, accuracies)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()


# 🧪 Step 7: Test new image
result = predict_image("test.jpg", mean, model, label_map)
print("Predicted Person:", result)