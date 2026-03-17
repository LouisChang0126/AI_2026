import librosa
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract audio features from a given file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Extract MFCC features
    zcr = librosa.feature.zero_crossing_rate(y) # Extract ZCR feature
    rms = librosa.feature.rms(y=y) # Extract RMS energy feature

    # Combine extracted features into a single feature vector
    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(zcr),
        np.std(zcr),
        np.mean(rms),
        np.std(rms)
    ])
    return feature_vector

def data_prepare(base_path="dataset2/"):
    # Initialize data storage
    data = []
    labels = []

    # class labels
    class_names = ["Drum_Solo", "Piano_Solo", "Violin_Solo", "Acoustic_Guitar_Solo", "Electric_Guitar_Solo"]
    # Loop through each instrument category
    for label, class_folder in enumerate(class_names):
        print(f"Extract features from {class_folder}...")
        class_path = os.path.join(base_path, class_folder)

        # Loop through each audio file in the category
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                file_path = os.path.join(class_path, file)

                # Extract features from the audio file
                features = extract_features(file_path)

                # Append features and corresponding label
                data.append(features)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    # Return data and labels without splitting here; cross validation handles splitting and scaling
    return data, labels

def decision_tree(X_train, X_test, y_train, y_test, max_depth=5, random_seed=123):
    # Train a Decision Tree classifier with a maximum depth
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)
    print(f"Growing tree with max depth = {max_depth}...")
    clf.fit(X_train, y_train)

    # Test the trained model on the test dataset
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Accuracy: {accuracy:.4f}")
    return accuracy

# plot the accuracy vs. max depth
def plot(accuracy):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy.keys(), accuracy.values(), marker='o', linestyle='-')

    plt.title("Decision Tree Accuracy vs. Max Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.xticks(list(accuracy.keys()))

    for key, value in accuracy.items():
        plt.text(key, value, f"{value:.4f}", ha='right', va='bottom')

    plt.grid(True)
    plt.savefig('decision_tree_depth.png')

if __name__ == "__main__":
    data, labels = data_prepare(base_path="dataset1/")  # dataset2/
    
    KFOLDS = 5
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=123)
    
    accuracy = {}
    for i in range(3, 24, 2):
        fold_accs = []
        print(f"--- Evaluating max_depth = {i} ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"Fold {fold+1}/{KFOLDS}")
            X_train, X_test = data[train_idx], data[val_idx]
            y_train, y_test = labels[train_idx], labels[val_idx]
            
            # Standardize the features to improve model performance
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            acc = decision_tree(X_train, X_test, y_train, y_test, max_depth=i)
            fold_accs.append(acc)
            
        avg_acc = np.mean(fold_accs)
        accuracy[i] = avg_acc
        print(f"Total Accuracy for max_depth={i}: {avg_acc:.4f}\n")
        
    plot(accuracy)
