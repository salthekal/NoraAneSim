##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

import numpy as np
import pandas as pd
import os
import pywt
import gc  # For manual garbage collection
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.fftpack import dct
from vmdpy import VMD
from scipy.fft import fft, dct  
from sklearn.decomposition import PCA
from scipy.fft import ifft
from pywt import waverec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-GUI environments
import matplotlib.pyplot as plt  # Import pyplot for plotting
from scipy.interpolate import interp1d




##########################################################################################################################
##########################################################################################################################
################################################# FUNCTION DEFINITIONS #######################################################################################
##########################################################################################################################
##########################################################################################################################


# Function to load EEG data from multiple CSV files
def load_eeg_data(base_dir, case_id):
    # Construct file paths
    print("Loading data...")
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
##########################################################################################################################
################################################# FUNCTION IMPLEMENTATION ################################################
##########################################################################################################################
##########################################################################################################################


# Example of applying processing across multiple case IDs
base_dir = '/home/ubuntu/NoraSim/Data/OR'
case_ids = [212, 222, 223, 228, 244]

all_dct_features = []
all_fft_features = []
feature_length = 250000  # Set a feature length based on your expected FFT size
all_band_powers = []
all_wavelet_features = []


for case_id in case_ids:
    sdb, frequencies, time, label = load_eeg_data(base_dir, case_id)

    # Check and adjust lengths
    if len(time) != len(sdb):
        # Option 1: Interpolate time data to match sdb length
        time_interpolator = interp1d(np.linspace(0, 1, len(time)), time)
        time = time_interpolator(np.linspace(0, 1, len(sdb)))

    # Create a function that interpolates 'label' to match the length of 'time'
    interpolator = interp1d(np.linspace(0, 1, len(label)), label, kind='nearest')  # 'nearest' can be changed based on data nature
    new_labels = interpolator(np.linspace(0, 1, len(time)))

    # Debug
    # Plotting Sdb as a function of Time
    plt.figure(figsize=(10, 4))
    plt.plot(time, sdb, label='Sdb')
    plt.title(f'Raw Sdb Over Time - Case ID {case_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Sdb Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Raw_Sdb_Over_Time_Case_ID_{case_id}.png')
    plt.close()

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

    #print(f"Calculated Frequency: {fs} Hz")

    # Preprocess: Detrending the signal
    #print("Detrending signal...")

    detrended_signal = detrend(sdb)

    #print(f"Signal detrended ")

    # Calculate PSD using Welch's method
    frequencies, psd = calculate_psd(detrended_signal, fs)

    # Debug
    # Additional plots within the loop
    plt.figure(figsize=(10, 4))
    plt.semilogy(frequencies, psd)
    plt.title(f'Power Spectral Density - Case ID {case_id}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.grid(True)
    plt.savefig(f'Power_Spectral_Density_Case_ID_{case_id}.png')
    plt.close()



    # Extract band powers from PSD
    band_powers = extract_band_powers(frequencies, psd)
    all_band_powers.append(band_powers)

    # Apply this function to your detrended signal

    vmd_signal = process_eeg_segments(detrended_signal)
    #print("Signal VMD")

    # Apply Denoising Methods
    ma_signal = moving_average(detrended_signal, window_size=10)
    #print("Signal MA")

    wavelet_signal = wavelet_denoising(detrended_signal)
    #print("Signal WL")

    gaussian_signal = apply_gaussian_filter(detrended_signal)
    #print("Gaussian filter")

    min_signal = apply_min_filter(detrended_signal)
    #print("Min.filter")
    min_max_signal = apply_min_max_filter(detrended_signal)
    #print("Max.filter")

    # Debug
    # Display denoising results
    plt.figure(figsize=(10, 6))
    plt.plot(detrended_signal, label='Original', alpha=0.5)
    plt.plot(ma_signal, label='Moving Average')
    plt.plot(wavelet_signal, label='Wavelet')
    plt.plot(gaussian_signal, label='Gaussian Filter')
    plt.plot(min_signal, label='Min Filter')
    plt.plot(min_max_signal, label='Min-Max Filter')
    plt.title(f'Denoising Effects - Case ID {case_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Denoising_Results_Case_ID_{case_id}.png')  # Include case_id in the filename
    plt.close()  # Close the plot to free up memory resources


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
    delta_band_signal = apply_bandpass_filter(detrended_signal, 0.5, 4, fs)  # fs needs to be adjusted if different

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

    # Debugging: Print shapes after loop
    #print(f"FFT Features Collected: {np.array(all_fft_features).shape}")
    #print(f"Wavelet Features Collected: {np.array(all_wavelet_features).shape}")
    #print(f"DCT Features Collected: {np.array(all_dct_features).shape}")
   # print(f"Band Powers Collected: {np.array(all_band_powers).shape}")

    # Debug
    # Feature plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(fft_features[0], label='FFT Features')
    plt.title(f'FFT Features - Case ID {case_id}')
    plt.xlabel('Feature Index')
    plt.ylabel('Magnitude')

    plt.subplot(1, 4, 2)
    if dct_features.size > 0:
        plt.plot(dct_features[0], label='DCT Features')
        plt.title(f'DCT Features - Case ID {case_id}')
    else:
        plt.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        
    
    plt.subplot(1, 4, 3)
    if w_features.size > 0:
        plt.plot(w_features[0], label='WL Features')
        plt.title(f'WL Features - Case ID {case_id}')
    else:
        plt.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    plt.subplot(1, 4, 4)
    if windowed_signal.size > 0:
        plt.plot(windowed_signal[0], label='Kaiser Signal')
        plt.title(f'Kaiser Signal - Case ID {case_id}')
    else:
        plt.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')


    plt.tight_layout()
    plt.savefig(f'FFT_DCT_WL_Kaiser_Features_Case_ID_{case_id}.png')
    plt.close()




# Feature redcution 
# Apply the aggregation function
fft_aggregated = aggregate_features(np.vstack(all_fft_features))
wavelet_aggregated = aggregate_features(np.vstack(all_wavelet_features))
dct_aggregated = aggregate_features(np.vstack(all_dct_features))


# Reconstruct signal from FFT features for SNR calculation
fft_reconstructed = ifft(fft_features).real
snr_fft = calculate_snr_reconstructed(detrended_signal, fft_reconstructed)

# Calculate SNR for an example case, adjust indices as needed
wavelet_reconstructed = wavelet_reconstruct(detrended_signal)

# Calculate SNR between the original detrended signal and the reconstructed signal
snr_wavelet_recon = calculate_snr_reconstructed(detrended_signal, wavelet_reconstructed)


# Reconstruct signal from DCT features for SNR calculation
dct_reconstructed = ifft(dct(fft_features, type=3)).real  # Inverse DCT
snr_dct = calculate_snr_reconstructed(detrended_signal, dct_reconstructed)

print(f"SNR for FFT reconstructed: {snr_fft}")
print(f"SNR for Wavelet reconstructed: {snr_wavelet_recon}")
print(f"SNR for DCT reconstructed: {snr_dct}")


#print("Reduced Welch Features Shape:", reduced_features.shape)
#print("Explained Welch Variance Ratio:", explained_variance)

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
def create_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Redefine and retrain the FFT features on CNN for the normalized data
cnn_model = create_cnn((X_fft_train.shape[1], 1))
cnn_history = cnn_model.fit(X_fft_train, y_train, epochs=10, validation_data=(X_fft_test, y_test), verbose=1)
cnn_accuracy = cnn_model.evaluate(X_fft_test, y_test, verbose=0)[1]
print(f'CNN FFT Test Accuracy: {cnn_accuracy}')


cnn_model_dct = create_cnn((X_dct_train.shape[1], 1))
cnn_history_dct = cnn_model.fit(X_dct_train, y_train, epochs=10, validation_data=(X_dct_test, y_test), verbose=1)
cnn_accuracy_dct = cnn_model.evaluate(X_dct_test, y_test, verbose=0)[1]
print(f'CNN DCT Test Accuracy: {cnn_accuracy_dct}')

cnn_model_wl = create_cnn((X_wl_train.shape[1], 1))
cnn_history_wl = cnn_model.fit(X_wl_train, y_train, epochs=10, validation_data=(X_wl_test, y_test), verbose=1)
cnn_accuracy_wl = cnn_model.evaluate(X_wl_test, y_test, verbose=0)[1]
print(f'CNN WL Test Accuracy: {cnn_accuracy_wl}')

# Define ANN model
def create_ann(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train ANN on FFT features
ann_model = create_ann(X_fft_train.shape[1])
ann_history = ann_model.fit(X_fft_train, y_train, epochs=10, validation_data=(X_fft_test, y_test), verbose=1)
ann_accuracy = ann_model.evaluate(X_fft_test, y_test, verbose=0)[1]
print(f'ANN FFT Test Accuracy: {ann_accuracy}')

# Train ANN on DCT features
ann_model_dct = create_ann(X_dct_train.shape[1])
ann_history_dct = ann_model.fit(X_dct_train, y_train, epochs=10, validation_data=(X_dct_test, y_test), verbose=1)
ann_accuracy_dct = ann_model.evaluate(X_dct_test, y_test, verbose=0)[1]
print(f'ANN DCT Test Accuracy: {ann_accuracy_dct}')

# Train ANN on WL features
ann_model_wl = create_ann(X_wl_train.shape[1])
ann_history_wl = ann_model.fit(X_wl_train, y_train, epochs=10, validation_data=(X_wl_test, y_test), verbose=1)
ann_accuracy_wl = ann_model.evaluate(X_wl_test, y_test, verbose=0)[1]
print(f'ANN WL Test Accuracy: {ann_accuracy_wl}')



def calculate_model_snr(accuracy):
    signal_power = accuracy ** 2
    noise_power = (1 - accuracy) ** 2 + 10e-10  # Add a small epsilon to avoid division by zero
    snr = 10 * np.log10(signal_power / noise_power)
    return snr



snr_cnn_fft = calculate_model_snr(cnn_accuracy)
snr_cnn_dct = calculate_model_snr(cnn_accuracy_dct)
snr_ann_fft = calculate_model_snr(ann_accuracy)
snr_ann_dct = calculate_model_snr(ann_accuracy_dct)
snr_cnn_wl = calculate_model_snr(cnn_accuracy_wl)
snr_ann_wl = calculate_model_snr(ann_accuracy_wl)

print(f'CNN FFT SNR: {snr_cnn_fft} dB')
print(f'CNN DCT SNR: {snr_cnn_dct} dB')
print(f'CNN WL SNR: {snr_cnn_wl} dB')
print(f'ANN FFT SNR: {snr_ann_fft} dB')
print(f'ANN DCT SNR: {snr_ann_dct} dB')
print(f'ANN WL SNR: {snr_ann_wl} dB')

##########################################################################################################################
##########################################################################################################################
################################################# VISULAIZATIONS  #######################################################################################
##########################################################################################################################
##########################################################################################################################



