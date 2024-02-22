import torch
import numpy as np
import pandas as pd
import random
from scipy.signal import butter, filtfilt, iirnotch
import torchvision.transforms as transforms
import multiprocessing
from torch.utils.data import DataLoader, Dataset
import matplotlib as mpl
from math import ceil
import argparse
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from scipy.signal import spectrogram
import pywt

fs = 2000 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 100 #50 ms
numElectrodes = 4
num_subjects = 40
cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_labels_partial = ['Rest', 'Extension', 'Flexion', 'Ulnar_Deviation', 'Radial_Deviation', 'Grip', 'Abduction'] 
gesture_labels_full = ['Rest', 'Extension', 'Flexion', 'Ulnar_Deviation', 'Radial_Deviation', 'Grip', 'Abduction', 'Adduction', 'Supination', 'Pronation']
gesture_labels = gesture_labels_full
numGestures = len(gesture_labels)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# not needed for Ozdemir
def balance (restimulus):
    numZero = 0
    indices = []
    #print(restimulus.shape)
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 380):
                    #print("working")
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), numGestures))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=[5.0, 500.0], btype='bandpass', analog=False, fs=2000.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=2000.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

# not needed for Ozdemir
def getRestim (n):
    assert n >= 1 and n <= num_subjects
    # read hdf5 file 
    restim = pd.read_hdf(f'DatasetsProcessed_hdf5/OzdemirEMG/p{n}/flattened_participant_{n}_E2.hdf5')
    restim = torch.tensor(restim.values)
    return restim.unfold(dimension=0, size=wLenTimesteps, step=stepLen)

def getEMG (n):
    assert n >= 1 and n <= num_subjects
    file = h5py.File(f'DatasetsProcessed_hdf5/OzdemirEMG/p{n}/flattened_participant_{n}.hdf5', 'r')
    emg = []
    for gesture in gesture_labels:
        assert "Gesture" + gesture in file, f"Gesture {gesture} not found in file for participant {n}!"
        data = filter(torch.from_numpy(np.array(file["Gesture" + gesture]))).unfold(dimension=-1, size=wLenTimesteps, step=stepLen)
        emg.append(torch.cat([data[i] for i in range(len(data))], dim=-2).permute((1, 0, 2)))
    return torch.cat(emg, dim=0)

# size of 4800 assumes 250 ms window
def getLabels (n):
    timesteps_for_one_gesture = 480
    labels = np.zeros((timesteps_for_one_gesture*numGestures, numGestures))
    for i in range(timesteps_for_one_gesture):
        for j in range(numGestures):
            labels[j * timesteps_for_one_gesture + i][j] = 1.0
    return labels

def optimized_makeOneCWTImage(data, length, width, resize_length_factor, native_resnet_size):
    emg_sample = data
    # Convert EMG sample to numpy array for CWT computation
    emg_sample_np = emg_sample.astype(np.float16).flatten()
    highest_cwt_scale = 31
    downsample_factor_for_cwt_preprocessing = 2 # used to make image processing tractable
    scales = np.arange(1, highest_cwt_scale)  
    wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet; adjust as needed
    # Perform Continuous Wavelet Transform (CWT)
    # Note: PyWavelets returns scales and coeffs (coefficients)
    coefficients, frequencies = pywt.cwt(emg_sample_np[::downsample_factor_for_cwt_preprocessing], scales, wavelet, sampling_period=1/fs)
    coefficients_dB = 10 * np.log10(np.abs(coefficients) + 1e-6)  # Adding a small constant to avoid log(0)
    # Convert back to PyTorch tensor and reshape
    emg_sample = torch.tensor(coefficients_dB).float().reshape(-1, coefficients_dB.shape[-1])
    # Normalization
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample) - torch.min(emg_sample)  # Adjusted normalization to avoid divide-by-zero
    blocks = emg_sample.reshape(highest_cwt_scale-1, numElectrodes, -1)
    emg_sample = blocks.transpose(1,0).reshape(numElectrodes*(highest_cwt_scale-1), -1)
        
    # Update 'window_size' if necessary
    window_size = emg_sample.shape[1]

    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    data = emg_sample

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))
    
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Since no split occurs, we don't need to concatenate halves back together
    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = 50
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    frequencies, times, Sxx = spectrogram(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size, noverlap=0.5*spectrogram_window_size)
    Sxx_dB = 10 * np.log10(Sxx + 1e-6) # small constant added to avoid log(0)
    emg_sample = torch.from_numpy(Sxx_dB)
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])
    data = emg_sample

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))
    
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Since no split occurs, we don't need to concatenate halves back together
    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOneMagnitudeImage(data, length, width, resize_length_factor, native_resnet_size, global_min, global_max):
    # Normalize with global min and max
    data = (data - global_min) / (global_max - global_min)
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    
    # Clamp between 0 and 1 using torch.clamp
    imageL, imageR = map(lambda img: torch.clamp(img, 0, 1), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def optimized_makeOneImage(data, cmap, length, width, resize_length_factor, native_resnet_size):
    # Contrast normalize and convert data
    # NOTE: Should this be contrast normalized? Then only patterns of data will be visible, not absolute values
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    
    # Get max and min values after interpolation
    max_val = max(imageL.max(), imageR.max())
    min_val = min(imageL.min(), imageR.min())
    
    # Contrast normalize again after interpolation
    imageL, imageR = map(lambda img: (img - min_val) / (max_val - min_val), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, turn_on_spectrogram=False, turn_on_cwt=False,
              global_min=None, global_max=None):

    emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        # Apply RMS calculation along the last axis (axis=-1)
        emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = emg_rms  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)

    # Parameters that don't change can be set once
    resize_length_factor = 6
    if turn_on_magnitude:
        resize_length_factor = 3
    native_resnet_size = 224

    with multiprocessing.Pool(processes=5) as pool:
        args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_async = pool.starmap_async(optimized_makeOneImage, args)
        images = images_async.get()

    if turn_on_magnitude:
        with multiprocessing.Pool(processes=5) as pool:
            args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
            images_async = pool.starmap_async(optimized_makeOneMagnitudeImage, args)
            images_magnitude = images_async.get()
        images = np.concatenate((images, images_magnitude), axis=2)

    elif turn_on_spectrogram:
        with multiprocessing.Pool(processes=5) as pool:
            args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
            images_async = pool.starmap_async(optimized_makeOneSpectrogramImage, args)
            images_spectrogram = images_async.get()
        images = images_spectrogram

    elif turn_on_cwt:
        with multiprocessing.Pool(processes=5) as pool:
            args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
            images_async = pool.starmap_async(optimized_makeOneCWTImage, args)
            images_cwt = images_async.get()
        images = images_cwt
        
    return images

def periodLengthForAnnealing(num_epochs, annealing_multiplier, cycles):
    periodLength = 0
    for i in range(cycles):
        periodLength += annealing_multiplier ** i
    periodLength = num_epochs / periodLength
    
    return ceil(periodLength)

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def plot_confusion_matrix(true, pred, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Calculate confusion matrix
    cf_matrix = confusion_matrix(true, pred)
    df_cm_unnormalized = pd.DataFrame(cf_matrix, index=gesture_labels, columns=gesture_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Plot confusion matrix square
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'{testrun_foldername}confusionMatrix_{partition_name}_seed{args.seed}_{formatted_datetime}.png'
    plt.savefig(confusionMatrix_filename)
    df_cm_unnormalized.to_pickle(f'{testrun_foldername}confusionMatrix_{partition_name}_seed{args.seed}_{formatted_datetime}.pkl')
    wandb.log({f"{partition_name} Confusion Matrix": wandb.Image(confusionMatrix_filename),
                f"Raw {partition_name.capitalize()} Confusion Matrix": wandb.Table(dataframe=df_cm_unnormalized)})
    
def denormalize(images):
    # Define mean and std from imageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    # Denormalize
    images = images * std + mean
    
    # Clip the values to ensure they are within [0,1] as expected for image data
    images = torch.clamp(images, 0, 1)
    
    return images


def plot_average_images(image_data, true, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Convert true to numpy for quick indexing
    true_np = np.array(true)        

    # Calculate average image of each gesture
    average_images = []
    print(f"Plotting average {partition_name} images...")
    for i in range(numGestures):
        # Find indices
        gesture_indices = np.where(true_np == i)[0]

        # Select and denormalize only the required images
        gesture_images = denormalize(transforms.Resize((224,224))(image_data[gesture_indices])).cpu().detach().numpy()
        average_images.append(np.mean(gesture_images, axis=0))

    average_images = np.array(average_images)
                    
    # Plot average image of each gesture
    fig, axs = plt.subplots(2, 9, figsize=(10, 10))
    for i in range(numGestures):
        axs[i//9, i%9].imshow(average_images[i].transpose(1,2,0))
        axs[i//9, i%9].set_title(gesture_labels[i])
        axs[i//9, i%9].axis('off')
    fig.suptitle('Average Image of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Log in wandb
    averageImages_filename = f'{testrun_foldername}averageImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(averageImages_filename, dpi=450)
    wandb.log({f"Average {partition_name.capitalize()} Images": wandb.Image(averageImages_filename)})


def plot_first_fifteen_images(image_data, true, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Convert true to numpy for quick indexing
    true_np = np.array(true)

    # Parameters for plotting
    rows_per_gesture = 15
    total_gestures = numGestures  # Replace with the actual number of gestures

    # Create subplots
    fig, axs = plt.subplots(rows_per_gesture, total_gestures, figsize=(20, 20))

    print(f"Plotting first fifteen {partition_name} images...")
    for i in range(total_gestures):
        # Find indices of the first 15 images for gesture i
        gesture_indices = np.where(true_np == i)[0][:rows_per_gesture]
        
        # Select and denormalize only the required images
        gesture_images = denormalize(transforms.Resize((224,224))(image_data[gesture_indices])).cpu().detach().numpy()

        for j in range(len(gesture_images)):  # len(gesture_images) is no more than rows_per_gesture
            ax = axs[j, i]
            # Transpose the image data to match the expected shape (H, W, C) for imshow
            ax.imshow(gesture_images[j].transpose(1, 2, 0))
            if j == 0:
                ax.set_title(gesture_labels[i])
            ax.axis('off')

    fig.suptitle(f'First Fifteen {partition_name.capitalize()} Images of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and log the figure
    firstThreeImages_filename = f'{testrun_foldername}firstFifteenImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(firstThreeImages_filename, dpi=300)
    wandb.log({f"First Fifteen {partition_name.capitalize()} Images of Each Gesture": wandb.Image(firstThreeImages_filename)})

