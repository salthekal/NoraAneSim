# NoraAneSim

This repository contains a Python script designed for comprehensive analysis of EEG data, focusing on the detection and classification of brain states related to anesthesia. The script utilizes advanced signal processing techniques, feature extraction, and machine learning models to analyze EEG signals and identify different states of consciousness.

Installation
Before running the script, ensure you have Python and the necessary libraries installed. You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas scipy pywavelets matplotlib tensorflow scikit-learn
Please note that you might need to install the vmdpy library separately. Instructions can typically be found in its official repository or documentation.

How to Use
Data Setup: Organize your EEG data into CSV files per case with specific naming conventions (`
_caseID__Sdb.csv, _caseID__f.csv, _caseID__t.csv, _caseID__l.csv`) for the signal, frequency, time, and label data respectively.
2. Configure the Script:

Set the base_dir variable in the script to the directory where your data files are stored.
Define the list of case_ids you want to analyze, e.g., [212, 222, 223, 228, 244].
Execution:
Run the script using Python from your command line:
bash
Copy code
python EEG_analysis_script.py
Features
Signal Denoising: Implements several techniques to clean the EEG signals from noise, including moving average, wavelet denoising, Gaussian smoothing, and minimum-maximum filtering.
Feature Extraction: Extracts features using Fourier Transform, Discrete Cosine Transform (DCT), Wavelet Transform, and band-pass filtering.
Feature Aggregation: Aggregates features to simplify the input into the machine learning models, enhancing the learning process.
Model Training and Evaluation: Trains Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) to classify different brain states from EEG data. Evaluates the models based on accuracy and computes the signal-to-noise ratio (SNR) for model performance assessment.
Visualization
The script generates several plots to visualize:

The effects of denoising.
Power Spectral Density (PSD) of the signals.
Extracted features for each transformation technique.
Ensure matplotlib is set to use the 'Agg' backend if you are running this in a non-GUI environment.
