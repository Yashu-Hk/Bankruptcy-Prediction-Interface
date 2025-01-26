import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the dataset path (update this if needed)
dataset_path = "../dataset/data.csv"

# Define the directory to save models
models_dir = "models"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path)

# Preprocess the dataset
print("Preprocessing dataset...")
y = data.iloc[:, 0]  # Target variable (1st column)
X = data.iloc[:, 1:]  # Features (all other columns)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

# Train and evaluate each model
print("Training models...")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Save the model
    model_filename = f"{models_dir}/{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    print(f"{model_name} saved to {model_filename}")

print("\nAll models trained and saved successfully!")