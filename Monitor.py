import numpy as np
import pandas as pd
import os
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend, welch
from scipy.fft import fft, dct
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function Definitions
def load_eeg_data(base_dir, case_id, patient_info):
    """ Load EEG data including patient demographics and conditions. """
    print(f"Loading data for case {case_id}, Patient Age: {patient_info['age']}, Condition: {patient_info['condition']}")
    file_paths = {ext: os.path.join(base_dir, f'{case_id}_{ext}.csv') for ext in ['Sdb', 'f', 't', 'l']}
    data = {ext: pd.read_csv(path, header=None).values.flatten() for ext, path in file_paths.items()}
    print("Data loaded")
    return data['Sdb'], data['f'], data['t'], data['l']

def extract_and_reduce_features(signal, method='fft', feature_length=250000, n_components=10, samples_needed=1):
    """Extract features using FFT, DCT, or Wavelet and reduce using PCA."""
    if method == 'fft':
        features = np.abs(fft(signal))[:feature_length]
    elif method == 'dct':
        features = dct(signal, type=2)[:feature_length]
    elif method == 'wavelet':
        coeffs = pywt.wavedec(signal, 'db4', level=5)
        features = np.concatenate([c.flatten() for c in coeffs])[:feature_length]
    else:
        return None, None

    if features.size > samples_needed:
        features = features[:samples_needed * (features.size // samples_needed)]
    features = features.reshape(-1, features.size // samples_needed)
    pca = PCA(n_components=min(n_components, features.shape[1]))
    reduced_features = pca.fit_transform(features)
    signal_power = np.mean(reduced_features**2)
    noise_power = np.mean((reduced_features - np.mean(reduced_features))**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    mse = mean_squared_error(np.zeros_like(reduced_features), reduced_features)
    return reduced_features, snr, mse

def create_cnn(input_shape, num_classes):
    """Create a CNN model for EEG signal classification."""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the CNN model and evaluate its performance."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = confusion_matrix(y_true, y_pred)[0, 0] / sum(confusion_matrix(y_true, y_pred)[0, :])
    specificity = confusion_matrix(y_true, y_pred)[1, 1] / sum(confusion_matrix(y_true, y_pred)[1, :])
    return history, accuracy, sensitivity, specificity

# Example Usage
base_dir = '/home/ubuntu/NoraSim/Data/OR'
case_ids = [62, 65, 132, 133, 143, 212, 222, 223, 228, 244]
patient_info = {'age': 35, 'condition': 'In surgery room under anesthesia'}

for case_id in case_ids:
    sdb, frequencies, time, label = load_eeg_data(base_dir, case_id, patient_info)
    signal_processed = detrend(sdb)
    fft_features, fft_snr, fft_mse = extract_and_reduce_features(signal_processed, 'fft', n_components=10, samples_needed=len(label))
    dct_features, dct_snr, dct_mse = extract_and_reduce_features(signal_processed, 'dct', n_components=10, samples_needed=len(label))
    wavelet_features, wavelet_snr, wavelet_mse = extract_and_reduce_features(signal_processed, 'wavelet', n_components=10, samples_needed=len(label))
    labels = to_categorical(np.where((label == 0) | (label == 1), label, 0), num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(fft_features, labels, test_size=0.3, random_state=42)
    model = create_cnn((fft_features.shape[1], 1), num_classes=2)
    history, accuracy, sensitivity, specificity = train_model(model, X_train, y_train, X_test, y_test)
    print(f"Results for FFT Features: SNR={fft_snr:.2f} dB, MSE={fft_mse:.2f}, Accuracy={accuracy:.4f}")
