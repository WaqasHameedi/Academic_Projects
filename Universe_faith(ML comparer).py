#Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
#dataset contain matter density , dark energy density and hubble constant
# Step 1: Define the Dataset
X = [
 [0.3, 0.7, 70], [0.5, 0.5, 60], [0.1, 0.9, 75], [0.4, 0.6, 65], [0.6, 0.4, 55],
 [0.35, 0.65, 68], [0.45, 0.55, 63], [0.25, 0.75, 72], [0.3, 0.7, 69], [0.2, 0.8, 74],
 [0.55, 0.45, 57], [0.1, 0.9, 76], [0.32, 0.68, 67], [0.42, 0.58, 64], [0.52, 0.48, 62],
 [0.37, 0.63, 66], [0.15, 0.85, 73], [0.6, 0.4, 54], [0.4, 0.6, 61], [0.33, 0.67, 71],
 [0.48, 0.52, 60], [0.22, 0.78, 70], [0.27, 0.73, 69], [0.31, 0.69, 65], [0.53, 0.47, 58],
 [0.36, 0.64, 62], [0.29, 0.71, 68], [0.4, 0.6, 67], [0.6, 0.4, 56], [0.38, 0.62, 66],
 [0.26, 0.74, 71], [0.3, 0.7, 64], [0.46, 0.54, 63], [0.2, 0.8, 75], [0.5, 0.5, 59],
 [0.35, 0.65, 70], [0.3, 0.7, 68], [0.42, 0.58, 60], [0.55, 0.45, 66], [0.25, 0.75, 72],
 [0.15, 0.85, 74], [0.48, 0.52, 61], [0.32, 0.68, 69], [0.6, 0.4, 57], [0.39, 0.61, 65],
 [0.4, 0.6, 62], [0.52, 0.48, 64], [0.45, 0.55, 60], [0.37, 0.63, 67], [0.2, 0.8, 73]
]
#faith 0 =big crunch,big freeze,big rip
y = [
 0, 1, 2, 0, 1, 0, 1, 2, 0, 2,
 1, 2, 0, 0, 1, 0, 2, 1, 1, 0,
 1, 2, 2, 0, 1, 0, 2, 0, 1, 0,
 2, 0, 1, 2, 1, 0, 0, 0, 1, 2,
 2, 0, 2, 1, 1, 2, 1, 0, 0, 2]

#new data from input
new_data=[]
new_data_entries =int (input(("enter how many new data entries you put:")))
for i in range(new_data_entries):
 print(f"\n entry{i +1}")
 matter_density=float(input("enter matter density: "))
 dark_energy_density=float(input("enter dark energy density: "))
 hubble_constant=float(input("enter hubble constant: "))
 new_data.append([matter_density,dark_energy_density,hubble_constant])
# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 3: Initialize classifiers
nb_model = GaussianNB()
logreg_model = LogisticRegression(max_iter=200)
knn_model = KNeighborsClassifier(n_neighbors=3)
# Step 4: Train classifiers
nb_model.fit(X_train, y_train)
logreg_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
# Step 5: Make predictions on test data and new data
y_pred_test_nb = nb_model.predict(X_test)
y_pred_new_nb = nb_model.predict(new_data)
y_pred_test_logreg = logreg_model.predict(X_test)
y_pred_new_logreg = logreg_model.predict(new_data)
y_pred_test_knn = knn_model.predict(X_test)
y_pred_new_knn = knn_model.predict(new_data)
# Step 6: Calculate accuracy scores
accuracy_nb = accuracy_score(y_test, y_pred_test_nb)
accuracy_logreg = accuracy_score(y_test, y_pred_test_logreg)
accuracy_knn = accuracy_score(y_test, y_pred_test_knn)
print("faith 0 =big crunch,1=big freeze,2=big rip")
print("Naive Bayes Accuracy:", accuracy_nb)
print("Logistic Regression Accuracy:", accuracy_logreg)
print("k-NN Accuracy:", accuracy_knn)
# Step 7: Plot results for each classifier
plt.figure(figsize=(14, 8))
# Naive Bayes plot
plt.subplot(3, 1, 1)
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual", marker="o")
plt.scatter(range(len(y_test)), y_pred_test_nb, color="red", label="Predicted (Naive Bayes)", marker="x")
plt.scatter(range(len(y_test), len(y_test) + len(new_data)), y_pred_new_nb, color="green", label="New Data (Naive Bayes)",
marker="*")
plt.title("Naive Bayes Classification")
plt.xlabel("Sample Index")
plt.ylabel("Fate Category")
plt.legend()
# Logistic Regression plot
plt.subplot(3, 1, 2)
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual", marker="o")
plt.scatter(range(len(y_test)), y_pred_test_logreg, color="red", label="Predicted (Logistic Regression)", marker="x")
plt.scatter(range(len(y_test), len(y_test) + len(new_data)), y_pred_new_logreg, color="green", label="New Data (Logistic Regression)", marker="*")
plt.title("Logistic Regression Classification")
plt.xlabel("Sample Index")
plt.ylabel("Fate Category")
plt.legend()
# k-NN plot
plt.subplot(3, 1, 3)
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual", marker="o")
plt.scatter(range(len(y_test)), y_pred_test_knn, color="red", label="Predicted (k-NN)", marker="x")
plt.scatter(range(len(y_test), len(y_test) + len(new_data)), y_pred_new_knn, color="green", label="New Data (k-NN)",
marker="*")
plt.title("k-NN Classification")
plt.xlabel("Sample Index")
plt.ylabel("Fate Category")
plt.legend()
plt.tight_layout()
plt.show()