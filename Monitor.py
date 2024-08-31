import numpy as np
import pandas as pd
import os
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
from scipy.fft import fft, dct
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function to load EEG data including demographics and conditions
def load_eeg_data(base_dir, case_id, patient_info):
    print(f"Loading data for case {case_id}, Patient Age: {patient_info['age']}, Condition: {patient_info['condition']}")
    file_paths = {ext: os.path.join(base_dir, f'{case_id}_{ext}.csv') for ext in ['Sdb', 'f', 't', 'l']}
    data = {ext: pd.read_csv(path, header=None).values.flatten() for ext, path in file_paths.items()}
    print("Data loaded")
    return data['Sdb'], data['f'], data['t'], data['l']

# Function to plot EEG signals
def plot_signals(time, signals, labels, title, filename):
    plt.figure(figsize=(10, 6))
    min_length = len(time)
    for signal, label in zip(signals, labels):
        adjusted_signal = signal[:min_length]
        plt.plot(time, adjusted_signal, label=label)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Function to plot extracted and reduced features
def plot_features(features, labels, title, filename):
    plt.figure(figsize=(12, 6))
    if len(features) > 1:
        for i, feature in enumerate(features):
            plt.subplot(1, len(features), i + 1)
            plt.plot(feature)
            plt.title(f'{labels[i]} Features')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Value')
    else:
        plt.plot(features[0])
        plt.title(title)
    plt.suptitle('Extracted Features Comparison')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to plot model training results
def plot_model_results(histories, labels, title, filename):
    plt.figure(figsize=(10, 5))
    for history, label in zip(histories, labels):
        plt.plot(history.history['accuracy'], label=f'{label} Train')
        plt.plot(history.history['val_accuracy'], label=f'{label} Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Function to create a CNN model for EEG signal classification
def create_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the CNN model and evaluate its performance
def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = confusion_matrix(y_true, y_pred)[0, 0] / sum(confusion_matrix(y_true, y_pred)[0, :])
    specificity = confusion_matrix(y_true, y_pred)[1, 1] / sum(confusion_matrix(y_true, y_pred)[1, :])
    return history, accuracy, sensitivity, specificity

# Main execution block
base_dir = '/home/ubuntu/NoraSim/Data/OR'
case_ids = [62, 65, 132, 133, 143, 212, 222, 223, 228, 244]
patient_info = {'age': 35, 'condition': 'In surgery room under anesthesia'}

histories = []
labels = []

for case_id in case_ids:
    sdb, frequencies, time, label = load_eeg_data(base_dir, case_id, patient_info)
    signal_processed = detrend(sdb)
    plot_signals(time, [sdb, signal_processed], ['Original', 'Detrended'], f'Signal Processing - Case ID {case_id}', f'signal_processing_{case_id}.png')
    
    fft_features, fft_snr, fft_mse = extract_and_reduce_features(signal_processed, 'fft', n_components=10, samples_needed=len(label))
    dct_features, dct_snr, dct_mse = extract_and_reduce_features(signal_processed, 'dct', n_components=10, samples_needed=len(label))
    wavelet_features, wavelet_snr, wavelet_mse = extract_and_reduce_features(signal_processed, 'wavelet', n_components=10, samples_needed=len(label))
    
    plot_features([fft_features, dct_features, wavelet_features], ['FFT', 'DCT', 'Wavelet'], 'Feature Extraction and Reduction', f'features_{case_id}.png')
    
    for features, label, feature_name in zip([fft_features, dct_features, wavelet_features], ['FFT', 'DCT', 'Wavelet']):
        labels = to_categorical(np.where((label == 0) | (label == 1), label, 0), num_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        model = create_cnn((features.shape[1], 1), num_classes=2)
        history = train_model(model, X_train, y_train, X_test, y_test)
        histories.append(history)
        labels.append(feature_name)

plot_model_results(histories, labels, 'Model Training Results', 'training_results.png')
