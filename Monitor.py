import numpy as np
import pandas as pd
import os
import pywt
import gc  # For manual garbage collection
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.fftpack import dct
from vmdpy import VMD
from scipy.fft import fft, ifft
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-GUI environments
import matplotlib.pyplot as plt  # Import pyplot for plotting
from scipy.interpolate import interp1d
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

##########################################################################################################################
################################################# FUNCTION DEFINITIONS #######################################################################################
##########################################################################################################################

# Function to load EEG data from multiple CSV files
def load_eeg_data(base_dir, case_id, patient_info=None):
    # Construct file paths
    print(f"Loading data for case {case_id}, Patient Info: {patient_info}")
    sdb_path = os.path.join(base_dir, f'{case_id}_Sdb.csv')
    frequencies_path = os.path.join(base_dir, f'{case_id}_f.csv')
    time_path = os.path.join(base_dir, f'{case_id}_t.csv')
    label_path = os.path.join(base_dir, f'{case_id}_l.csv')

    # Load data from CSV files, assuming no header and only one column
    sdb = pd.read_csv(sdb_path, header=None).values.flatten()
    frequencies = pd.read_csv(frequencies_path, header=None).values.flatten()
    time = pd.read_csv(time_path, header=None).values.flatten()
    label = pd.read_csv(label_path, header=None).values.flatten()

    print(f"Data loaded")

    return sdb, frequencies, time, label

##########################################################################################################################
################################################# DENOISING FUNCTIONS ###################################################
##########################################################################################################################

# Define denoising functions
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

def wavelet_denoising(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = pywt.threshold(coeffs[-1], np.std(coeffs[-1])/2, mode='soft')
    coeffs[-1] = np.where(np.abs(coeffs[-1]) < threshold, 0, coeffs[-1])
    return pywt.waverec(coeffs, wavelet)

def apply_gaussian_filter(signal, sigma=1):
    return gaussian_filter(signal, sigma=sigma)

def apply_min_filter(signal, size=3):
    return minimum_filter(signal, size=size)

def apply_min_max_filter(signal, size=3):
    return (minimum_filter(signal, size=size) + maximum_filter(signal, size=size)) / 2

def perform_vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-6):
    # Reduced K and increased tolerance
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return np.sum(u, axis=0)

##########################################################################################################################
################################################# SNR FUNCTIONS ##########################################################
##########################################################################################################################

# Helper function to calculate SNR for reconstructed signals
def calculate_snr_reconstructed(original, reconstructed):
    # Ensure length matching, in case the reconstructed signal has edge artifacts
    min_length = min(len(original), len(reconstructed))
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]

    noise = original - reconstructed
    power_signal = np.mean(original**2)
    power_noise = np.mean(noise**2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

# Example function for wavelet reconstruction
def wavelet_reconstruct(signal):
    coeffs = pywt.wavedec(signal, 'db4', mode='symmetric')
    reconstructed_signal = pywt.waverec(coeffs, 'db4', mode='symmetric')
    return reconstructed_signal

def calculate_snr(original, denoised):
    noise = original[:len(denoised)] - denoised
    power_signal = np.mean(original[:len(denoised)] ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / power_noise)

##########################################################################################################################
################################################# SEGMENTATION FUNCTIONS ##########################################################
##########################################################################################################################

def process_eeg_segments(signal, segment_size=2500, batch_size=10):  # Added batch processing
    segments = [signal[i:i + segment_size] for i in range(0, len(signal), segment_size)]
    vmd_signals = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        for segment in batch:
            try:
                vmd_signal = perform_vmd(segment)
                vmd_signals.append(vmd_signal)
            except MemoryError as e:
                print(f"Failed to process segment due to memory error: {e}")
                continue

        # Memory management
        del batch
        gc.collect()

    return np.concatenate(vmd_signals) if vmd_signals else np.array([])

##########################################################################################################################
################################################# FEATURE EXTRACTION FUNCTIONS ##########################################################
##########################################################################################################################

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def wavelet_features(signal, wavelet='db4', max_level=5):
    # Perform wavelet decomposition at specified level
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    # Flatten the list of coefficients
    features = np.concatenate([c.flatten() for c in coeffs])
    # Optionally truncate or pad features to ensure uniformity
    desired_length = 250000  # Adjust as needed
    if len(features) > desired_length:
        return features[:desired_length]
    elif len(features) < desired_length:
        return np.pad(features, (0, desired_length - len(features)), 'constant')
    return features

def discrete_cosine_transform(signal, feature_length=250000):
    dct_result = dct(signal, type=2)
    if len(dct_result) > feature_length:
        return dct_result[:feature_length]  # Truncate
    elif len(dct_result) < feature_length:
        return np.pad(dct_result, (0, feature_length - len(dct_result)), 'constant')  # Pad
    return dct_result

def apply_kaiser_window(signal, beta=14):
    window = np.kaiser(len(signal), beta)
    return signal * window

def fourier_features(signal, feature_length=250000):  # Define a suitable feature length based on your data
    fft_result = fft(signal)
    fft_magnitudes = np.abs(fft_result)
    pad_length = feature_length - len(fft_magnitudes)
    if pad_length > 0:
        # Zero-padding if the feature is shorter than the desired length
        fft_magnitudes = np.pad(fft_magnitudes, (0, pad_length), 'constant')
    else:
        # Truncate if the feature is longer than the desired length
        fft_magnitudes = fft_magnitudes[:feature_length]
    return fft_magnitudes.reshape(1, -1)  # Reshape for PCA

##########################################################################################################################
################################################# FEATURE REDUCTION FUNCTIONS ############################################
##########################################################################################################################

def calculate_psd(signal, fs):
    # Calculate Power Spectral Density using Welch's method
    frequencies, psd = welch(signal, fs, nperseg=1024)
    return frequencies, psd

def extract_band_powers(frequencies, psd):
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (13, 30), 'gamma': (30, 100)}
    band_powers = np.array([np.sum(psd[(frequencies >= low) & (frequencies <= high)]) for _, (low, high) in bands.items()])
    return band_powers.reshape(1, -1)

# Helper function to aggregate features
def aggregate_features(feature_matrix):
    """Calculate mean, std, min, and max for each feature set."""
    mean_features = np.mean(feature_matrix, axis=0)
    std_features = np.std(feature_matrix, axis=0)
    min_features = np.min(feature_matrix, axis=0)
    max_features = np.max(feature_matrix, axis=0)
    return np.hstack([mean_features, std_features, min_features, max_features])

##########################################################################################################################
################################################# VISUALIZATION FUNCTIONS ################################################
##########################################################################################################################

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

# Function to detrend a signal and extract trend parameters
def detrend_signal(signal, time):
    time = np.array(time).reshape(-1, 1)
    signal = np.array(signal).reshape(-1, 1)
    model = LinearRegression()
    model.fit(time, signal)
    beta_0 = model.intercept_[0]
    beta_1 = model.coef_[0][0]

    trend = beta_0 + beta_1 * time
    detrended_signal = signal - trend

    print(f"Detrending Parameters: Beta_0 (Intercept) = {beta_0}, Beta_1 (Slope) = {beta_1}")
    return detrended_signal.flatten(), beta_0, beta_1

# Function to extract time-domain features
def extract_time_domain_features(signal):
    features = {
        'Mean': np.mean(signal),
        'Variance': np.var(signal),
        'Amplitude': np.max(signal) - np.min(signal),
        'Skewness': stats.skew(signal),
        'Kurtosis': stats.kurtosis(signal)
    }
    return features

# Function to extract frequency-domain features using FFT
def extract_frequency_domain_features(signal, sampling_rate=250):
    freqs = fft(signal)
    power = np.abs(freqs) ** 2
    total_power = np.sum(power)
    peak_frequency = np.argmax(power) * sampling_rate / len(signal)
    bandwidth = np.sum(power > 0.5 * np.max(power))
    entropy = -np.sum(power * np.log2(power + np.finfo(float).eps))

    features = {
        'Total Power': total_power,
        'Peak Frequency': peak_frequency,
        'Bandwidth': bandwidth,
        'Entropy': entropy
    }
    return features

# Function to apply ICA and extract components
def apply_ica(signal, n_components):
    ica = FastICA(n_components=n_components, random_state=42)
    components = ica.fit_transform(signal.reshape(-1, 1))
    print(f"ICA Components extracted: {components.shape[1]}")
    return components

##########################################################################################################################
##########################################################################################################################
################################################# FUNCTION IMPLEMENTATION ################################################
##########################################################################################################################
##########################################################################################################################

# Example of applying processing across multiple case IDs
base_dir = '/home/ubuntu/NoraSim/Data/OR'
case_ids = [212, 222, 223, 228, 244]
patient_info = {'age': 35, 'condition': 'In surgery room under anesthesia'}

all_dct_features = []
all_fft_features = []
feature_length = 250000  # Set a feature length based on your expected FFT size
all_band_powers = []
all_wavelet_features = []
histories = []
labels_list = []

for case_id in case_ids:
    sdb, frequencies, time, label = load_eeg_data(base_dir, case_id, patient_info)

    # Check and adjust lengths
    if len(time) != len(sdb):
        # Option 1: Interpolate time data to match sdb length
        time_interpolator = interp1d(np.linspace(0, 1, len(time)), time)
        time = time_interpolator(np.linspace(0, 1, len(sdb)))

    # Create a function that interpolates 'label' to match the length of 'time'
    interpolator = interp1d(np.linspace(0, 1, len(label)), label, kind='nearest')  # 'nearest' can be changed based on data nature
    new_labels = interpolator(np.linspace(0, 1, len(time)))

    # Plotting Sdb as a function of Time
    plot_signals(time, [sdb], ['Sdb'], f'Raw Sdb Over Time - Case ID {case_id}', f'Raw_Sdb_Over_Time_Case_ID_{case_id}.png')

    # Plotting labels as a function of Time with color coding
    plt.figure(figsize=(10, 4))

    # Split the data based on labels for different coloring
    time_conscious = time[new_labels == 1]
    time_unconscious = time[new_labels == 0]
    labels_conscious = new_labels[new_labels == 1]
    labels_unconscious = new_labels[new_labels == 0]

    # Plot each group with different colors and labels
    plt.scatter(time_unconscious, labels_unconscious, color='red', label='Unconscious')
    plt.scatter(time_conscious, labels_conscious, color='blue', label='Conscious')

    plt.title(f'Labels Over Time - Case ID {case_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Consciousness Label')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Labels_Over_Time_Case_ID_{case_id}.png')
    plt.close()

    # Calculate the sampling frequency based on the frequency data
    fs = 2 * max(frequencies)  # Assume max frequency is the highest in the frequencies array

    # Detrending the signal
    detrended_signal, beta_0, beta_1 = detrend_signal(sdb, time)
    plot_signals(time, [sdb, detrended_signal], ['Original', 'Detrended'], f'Signal Processing - Case ID {case_id}', f'signal_processing_{case_id}.png')

    # Extracting time-domain features
    time_features = extract_time_domain_features(detrended_signal)

    # Calculate PSD using Welch's method
    frequencies_psd, psd = calculate_psd(detrended_signal, fs)

    # Plotting Power Spectral Density
    plt.figure(figsize=(10, 4))
    plt.semilogy(frequencies_psd, psd)
    plt.title(f'Power Spectral Density - Case ID {case_id}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.grid(True)
    plt.savefig(f'Power_Spectral_Density_Case_ID_{case_id}.png')
    plt.close()

    # Extract band powers from PSD
    band_powers = extract_band_powers(frequencies_psd, psd)
    all_band_powers.append(band_powers.flatten())

    # Apply Denoising Methods
    ma_signal = moving_average(detrended_signal, window_size=10)
    wavelet_signal = wavelet_denoising(detrended_signal)
    gaussian_signal = apply_gaussian_filter(detrended_signal)
    min_signal = apply_min_filter(detrended_signal)
    min_max_signal = apply_min_max_filter(detrended_signal)
    vmd_signal = process_eeg_segments(detrended_signal)

    # Plotting denoising results
    plot_signals(time[:len(ma_signal)], [detrended_signal[:len(ma_signal)], ma_signal], ['Original', 'Moving Average'], f'Denoising - Moving Average - Case ID {case_id}', f'Denoising_MA_Case_ID_{case_id}.png')
    plot_signals(time[:len(wavelet_signal)], [detrended_signal[:len(wavelet_signal)], wavelet_signal], ['Original', 'Wavelet'], f'Denoising - Wavelet - Case ID {case_id}', f'Denoising_Wavelet_Case_ID_{case_id}.png')

    # Evaluate the effect of each denoising method
    snr_ma = calculate_snr(detrended_signal, ma_signal)
    snr_wavelet = calculate_snr(detrended_signal, wavelet_signal)
    snr_gaussian = calculate_snr(detrended_signal, gaussian_signal)
    snr_min = calculate_snr(detrended_signal, min_signal)
    snr_min_max = calculate_snr(detrended_signal, min_max_signal)
    snr_vmd = calculate_snr(detrended_signal, vmd_signal)

    # Display SNR results
    print(f"SNR for Moving Average: {snr_ma}")
    print(f"SNR for Wavelet: {snr_wavelet}")
    print(f"SNR for Gaussian: {snr_gaussian}")
    print(f"SNR for Minimum Filter: {snr_min}")
    print(f"SNR for Min-Max Filter: {snr_min_max}")
    print(f"SNR for VMD: {snr_vmd}")

    # Feature Extraction
    # Focusing on Delta band (0.5-4 Hz) relevant for anesthesia research
    delta_band_signal = apply_bandpass_filter(detrended_signal, 0.5, 4, fs)

    # Apply Kaiser Window
    windowed_signal = apply_kaiser_window(delta_band_signal)

    # Feature Extraction
    fft_features = fourier_features(windowed_signal, feature_length)
    w_features = wavelet_features(detrended_signal)
    dct_features = discrete_cosine_transform(detrended_signal, feature_length=250000)

    # Appending to the list
    all_fft_features.append(fft_features.flatten())
    all_wavelet_features.append(w_features.flatten())
    all_dct_features.append(dct_features.flatten())
    all_band_powers.append(band_powers.flatten())  # Flatten if necessary

    # Plot features
    plot_features([fft_features.flatten(), dct_features.flatten(), w_features.flatten()], ['FFT', 'DCT', 'Wavelet'], f'Feature Comparison - Case ID {case_id}', f'Features_Case_ID_{case_id}.png')

    # Feature Reduction using PCA
    all_features = np.vstack([fft_features.flatten(), dct_features.flatten(), w_features.flatten()]).T
    reduced_features, explained_variance = apply_pca(all_features, n_components=10)

    # Prepare labels for training (assuming binary classification)
    y_label = np.zeros(reduced_features.shape[0])
    y_label[:reduced_features.shape[0] // 2] = 1  # Dummy labels, adjust as per actual labels
    y_label = to_categorical(y_label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(reduced_features, y_label, test_size=0.2, random_state=42)

    # CNN Model
    input_shape = (X_train.shape[1], 1)
    cnn_model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("CNN Structure:")
    cnn_model.summary()

    # Reshape data for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the model
    history = cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=10, verbose=1)
    histories.append(history)
    labels_list.append(f'Case ID {case_id}')

    # Evaluate the model
    predictions = cnn_model.predict(X_test_cnn)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'CNN Accuracy for Case ID {case_id}: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# Plot model training results
plot_model_results(histories, labels_list, 'Model Training Results', 'training_results.png')

##########################################################################################################################
##########################################################################################################################
################################################# CLASSIFICATION AND MODELS #######################################################################################
##########################################################################################################################
##########################################################################################################################

# Convert lists to numpy arrays after the loop
X_fft = np.vstack(all_fft_features)
X_wl = np.vstack(all_wavelet_features)
X_dct = np.vstack(all_dct_features)
X_band_powers = np.vstack(all_band_powers)
labels = np.random.randint(2, size=len(X_fft))  # Create a label for each feature set
y = to_categorical(labels)

# Perform train-test split
X_fft_train, X_fft_test, y_train, y_test = train_test_split(X_fft, y, test_size=0.2, random_state=42)
X_dct_train, X_dct_test, _, _ = train_test_split(X_dct, y, test_size=0.2, random_state=42)
X_wl_train, X_wl_test, _, _ = train_test_split(X_wl, y, test_size=0.2, random_state=42)

# Define model building function
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Redefine and retrain the FFT features on CNN for the normalized data
cnn_model_fft = create_cnn_model((X_fft_train.shape[1], 1), y_train.shape[1])
cnn_history_fft = cnn_model_fft.fit(X_fft_train.reshape(-1, X_fft_train.shape[1], 1), y_train, epochs=10, validation_data=(X_fft_test.reshape(-1, X_fft_test.shape[1], 1), y_test), verbose=1)
cnn_accuracy_fft = cnn_model_fft.evaluate(X_fft_test.reshape(-1, X_fft_test.shape[1], 1), y_test, verbose=0)[1]
print(f'CNN FFT Test Accuracy: {cnn_accuracy_fft}')

cnn_model_dct = create_cnn_model((X_dct_train.shape[1], 1), y_train.shape[1])
cnn_history_dct = cnn_model_dct.fit(X_dct_train.reshape(-1, X_dct_train.shape[1], 1), y_train, epochs=10, validation_data=(X_dct_test.reshape(-1, X_dct_test.shape[1], 1), y_test), verbose=1)
cnn_accuracy_dct = cnn_model_dct.evaluate(X_dct_test.reshape(-1, X_dct_test.shape[1], 1), y_test, verbose=0)[1]
print(f'CNN DCT Test Accuracy: {cnn_accuracy_dct}')

cnn_model_wl = create_cnn_model((X_wl_train.shape[1], 1), y_train.shape[1])
cnn_history_wl = cnn_model_wl.fit(X_wl_train.reshape(-1, X_wl_train.shape[1], 1), y_train, epochs=10, validation_data=(X_wl_test.reshape(-1, X_wl_test.shape[1], 1), y_test), verbose=1)
cnn_accuracy_wl = cnn_model_wl.evaluate(X_wl_test.reshape(-1, X_wl_test.shape[1], 1), y_test, verbose=0)[1]
print(f'CNN WL Test Accuracy: {cnn_accuracy_wl}')

# Define ANN model
def create_ann(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train ANN on FFT features
ann_model_fft = create_ann(X_fft_train.shape[1], y_train.shape[1])
ann_history_fft = ann_model_fft.fit(X_fft_train, y_train, epochs=10, validation_data=(X_fft_test, y_test), verbose=1)
ann_accuracy_fft = ann_model_fft.evaluate(X_fft_test, y_test, verbose=0)[1]
print(f'ANN FFT Test Accuracy: {ann_accuracy_fft}')

# Train ANN on DCT features
ann_model_dct = create_ann(X_dct_train.shape[1], y_train.shape[1])
ann_history_dct = ann_model_dct.fit(X_dct_train, y_train, epochs=10, validation_data=(X_dct_test, y_test), verbose=1)
ann_accuracy_dct = ann_model_dct.evaluate(X_dct_test, y_test, verbose=0)[1]
print(f'ANN DCT Test Accuracy: {ann_accuracy_dct}')

# Train ANN on WL features
ann_model_wl = create_ann(X_wl_train.shape[1], y_train.shape[1])
ann_history_wl = ann_model_wl.fit(X_wl_train, y_train, epochs=10, validation_data=(X_wl_test, y_test), verbose=1)
ann_accuracy_wl = ann_model_wl.evaluate(X_wl_test, y_test, verbose=0)[1]
print(f'ANN WL Test Accuracy: {ann_accuracy_wl}')

# Calculate SNR for models
def calculate_model_snr(accuracy):
    signal_power = accuracy ** 2
    noise_power = (1 - accuracy) ** 2 + 1e-10  # Add a small epsilon to avoid division by zero
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_cnn_fft = calculate_model_snr(cnn_accuracy_fft)
snr_cnn_dct = calculate_model_snr(cnn_accuracy_dct)
snr_ann_fft = calculate_model_snr(ann_accuracy_fft)
snr_ann_dct = calculate_model_snr(ann_accuracy_dct)
snr_cnn_wl = calculate_model_snr(cnn_accuracy_wl)
snr_ann_wl = calculate_model_snr(ann_accuracy_wl)

print(f'CNN FFT SNR: {snr_cnn_fft} dB')
print(f'CNN DCT SNR: {snr_cnn_dct} dB')
print(f'CNN WL SNR: {snr_cnn_wl} dB')
print(f'ANN FFT SNR: {snr_ann_fft} dB')
print(f'ANN DCT SNR: {snr_ann_dct} dB')
print(f'ANN WL SNR: {snr_ann_wl} dB')
