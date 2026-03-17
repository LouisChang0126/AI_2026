import librosa
import numpy as np
import os
from sklearn.svm import SVC
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

def svm_model(X_train, X_test, y_train, y_test, C=1.0, random_seed=123):
    # Train a Support Vector Machine classifier with a specific C
    clf = SVC(C=C, kernel='rbf', random_state=random_seed)
    print(f"Training SVM with C = {C}...")
    clf.fit(X_train, y_train)

    # Test the trained model on the test dataset
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Accuracy: {accuracy:.4f}")
    return accuracy

# plot the accuracy vs. C parameter
def plot(accuracy):
    plt.figure(figsize=(10, 5))
    
    # Extract keys and values
    c_values = list(accuracy.keys())
    c_labels = [str(c) for c in c_values]
    acc_values = list(accuracy.values())
    
    # Use semilogx since C values are usually on a logarithmic scale
    plt.semilogx(c_values, acc_values, marker='o', linestyle='-')

    plt.title("SVM Accuracy vs. C (Penalty Parameter)")
    plt.xlabel("C Parameter (log scale)")
    plt.ylabel("Accuracy")
    plt.xticks(c_values, c_labels)

    for key, value in accuracy.items():
        plt.text(key, value, f"{value:.4f}", ha='right', va='bottom')

    plt.grid(True)
    plt.savefig('svm_c.png')

if __name__ == "__main__":
    data, labels = data_prepare(base_path="dataset/")  # dataset2/
    
    KFOLDS = 5
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=123)
    
    accuracy = {}
    
    # A typical list of C values to test
    c_list = [0.01, 0.1, 1, 10, 100, 1000]
    
    for c in c_list:
        fold_accs = []
        print(f"--- Evaluating C = {c} ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"Fold {fold+1}/{KFOLDS}")
            X_train, X_test = data[train_idx], data[val_idx]
            y_train, y_test = labels[train_idx], labels[val_idx]
            
            # Standardize the features to improve model performance
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            acc = svm_model(X_train, X_test, y_train, y_test, C=c)
            fold_accs.append(acc)
            
        avg_acc = np.mean(fold_accs)
        accuracy[c] = avg_acc
        print(f"Total Accuracy for C={c}: {avg_acc:.4f}\n")
        
    plot(accuracy)
