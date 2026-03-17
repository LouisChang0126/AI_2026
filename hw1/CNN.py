import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
SAMPLE_RATE = 22050
N_MELS = 128
BATCH_SIZE = 8 # 8, 16, 32, 64
EPOCHS = 10
LEARNING_RATE = 0.001
KFOLDS = 5
BASE_PATH = "dataset/" # dataset2/

# Random seeds
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_mel_spectrogram(file_path):
    # Extracts mel spectrogram from an audio file and convert to decibel scale.
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

class AudioDataset(Dataset):
    # Custom PyTorch dataset for loading and processing audio files.
    def __init__(self, base_path, class_names):
        self.data = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}  # Map class names to indices

        # Load data and labels
        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(class_path, file)

                    # Extract features from the audio file
                    mel_spec = extract_mel_spectrogram(file_path)

                    # Append features and corresponding label
                    self.data.append(mel_spec)
                    self.labels.append(self.class_to_idx[class_name])
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Dataset loaded: {len(self.data)} samples")

        # Define transformation for input data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec = self.transform(self.data[idx])  # Apply transformation
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label

class AudioCNN(nn.Module):
    # CNN for audio classification.
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 53, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_confusion_matrix(y_true, y_pred, class_names):
    # Plots a confusion matrix for predictions.
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    # Dataset and class labels
    base_path = BASE_PATH
    class_names = ["Drum_Solo", "Piano_Solo", "Violin_Solo", "Acoustic_Guitar_Solo", "Electric_Guitar_Solo"]
    dataset = AudioDataset(base_path, class_names)
    accs = []
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=123)

    # Cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{KFOLDS}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = AudioCNN(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        plot_confusion_matrix(y_true, y_pred, class_names)
    
    print(f"Total Accuracy: {np.mean(accs):.4f}")
