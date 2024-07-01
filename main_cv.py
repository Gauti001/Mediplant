import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import joblib

def extract_hog_features(image):
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten()

# Load images and extract HOG features
def load_data(data_dir):
    data = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)

        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is not None:
                    hog_features = extract_hog_features(image)
                    data.append(hog_features)
                    labels.append(label)

    return np.array(data), np.array(labels)

# Load data
data, labels = load_data("Database_old")

# Feature normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Class balancing
smote = SMOTE(random_state=42)
data_balanced, labels_balanced = smote.fit_resample(data_scaled, labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_balanced, labels_balanced, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVM
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(max_iter=10000), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

# Train the best model
best_clf.fit(X_train, y_train)

# Predictions
y_pred = best_clf.predict(X_test)

# Model evaluation
class_report = classification_report(y_test, y_pred, target_names=np.unique(labels))
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Save the model
model_filename = "svm_model_updated_v4.pkl"
joblib.dump(best_clf, model_filename)
#MODIFIED 95%CODE