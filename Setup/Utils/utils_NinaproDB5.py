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
from tqdm.contrib.concurrent import process_map  # Use process_map from tqdm.contrib
from scipy.signal import spectrogram, stft
import pywt
import fcwt
import emd

fs = 200 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 50 #50 ms
stepLen = int(stepLen / 1000 * fs)
numElectrodes = 16
num_subjects = 10
cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_labels = {}
gesture_labels['Rest'] = ['Rest'] # Shared between exercises

gesture_labels[1] = ['Index Flexion', 'Index Extension', 'Middle Flexion', 'Middle Extension', 'Ring Flexion', 'Ring Extension',
                    'Little Finger Flexion', 'Little Finger Extension', 'Thumb Adduction', 'Thumb Abduction', 'Thumb Flexion',
                    'Thumb Extension'] # End exercise A

gesture_labels[2] = ['Thumb Up', 'Index Middle Extension', 'Ring Little Flexion', 'Thumb Opposition', 'Finger Abduction', 'Fist', 'Pointing Index', 'Finger Adduction',
                    'Middle Axis Supination', 'Middle Axis Pronation', 'Little Axis Supination', 'Little Axis Pronation', 'Wrist Flexion', 'Wrist Extension', 'Radial Deviation',
                    'Ulnar Deviation', 'Wrist Extension Fist'] # End exercise B

gesture_labels[3] = ['Large Diameter Grasp', 'Small Diameter Grasp', 'Fixed Hook Grasp', 'Index Finger Extension Grasp', 'Medium Wrap',
                    'Ring Grasp', 'Prismatic Four Fingers Grasp', 'Stick Grasp', 'Writing Tripod Grasp', 'Power Sphere Grasp', 'Three Finger Sphere Grasp', 'Precision Sphere Grasp',
                    'Tripod Grasp', 'Prismatic Pinch Grasp', 'Tip Pinch Grasp', 'Quadrupod Grasp', 'Lateral Grasp', 'Parallel Extension Grasp', 'Extension Type Grasp', 'Power Disk Grasp',
                    'Open A Bottle With A Tripod Grasp', 'Turn A Screw', 'Cut Something'] # End exercise C

partial_gesture_labels = ['Rest', 'Finger Abduction', 'Fist', 'Finger Adduction', 'Middle Axis Supination', 
                          'Middle Axis Pronation', 'Wrist Flexion', 'Wrist Extension', 'Radial Deviation', 'Ulnar Deviation']
partial_gesture_indices = [0] + [gesture_labels[2].index(g) + len(gesture_labels['Rest']) for g in partial_gesture_labels[1:]] # 0 is for rest 
transition_labels = ['Not a Transition', 'Transition']

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


def balance_gesture_classifier(restimulus, args):
    """ Balances distribution of restimulus by minimizing zero (rest) gestures.

    Args:
        restimulus (tensor): restimulus tensor
        args: argument parser object

    """
    numZero = 0
    indices = []
    count_dict = {}
    
    # First pass: count the occurrences of each unique tensor
    for x in range(len(restimulus)):
        unique_elements = torch.unique(restimulus[x])
        if len(unique_elements) == 1:
            element = unique_elements.item()
            element = (element, )
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1

        else:
            if args.include_transitions:
                elements = (restimulus[x][0][0].item(), restimulus[x][0][-1].item()) # take first and last gesture (transition window)

                if elements in count_dict:
                    count_dict[elements] += 1
                else:
                    count_dict[elements] = 1
                
    # Calculate average count of non-zero elements
    non_zero_counts = [count for key, count in count_dict.items() if key != (0,)]
    if non_zero_counts:
        avg_count = sum(non_zero_counts) / len(non_zero_counts)
    else:
        avg_count = 0  # Handle case where there are no non-zero unique elements

    for x in range(len(restimulus)):
        unique_elements = torch.unique(restimulus[x])
        if len(unique_elements) == 1:
            gesture = unique_elements.item()
            if gesture == 0: 
                if numZero < avg_count:
                    indices.append(x)
                numZero += 1 # Rest always in partial
            else:
                indices.append(x)
        else:
            if args.include_transitions:
                indices.append(x)
    return indices

def balance_transition_classifier(restimulus, args):
    '''
    Balances such that there is an equal number of windows for all types of gestures. Balances all combinations of (start_gesture, end_gesture) windows not just between transition and non transition. 
    '''
    indices = []

    transition_total = 0 
    non_transition_total = 0 

    transition_seen = {}
    non_transition_seen = {}
    
    # First pass to count the number of each type of window
    for x in range(len(restimulus)):

        start_gesture = restimulus[x][0][0].item()
        end_gesture = restimulus[x][0][-1].item()
        gesture = (start_gesture, end_gesture)

        if start_gesture == end_gesture: 
            non_transition_seen[gesture] = non_transition_seen.get(gesture, 0) + 1
            non_transition_total += 1

        else: 
            transition_seen[gesture] = transition_seen.get(gesture, 0) + 1
            transition_total += 1

    # Calculate average count of each gesture -- averaged seperately for transition and non-transition gestures
    equal_threshold = min(transition_total, non_transition_total)
    equal_transition_threshold = equal_threshold // len(transition_seen)
    equal_non_transition_threshold = equal_threshold // len(non_transition_seen)

    non_transition_windows_left = {key: equal_non_transition_threshold for key in non_transition_seen}
    transition_windows_left = {key: equal_transition_threshold for key in transition_seen}

    # Second pass: Add transtion/non-transition windows balanced per gesture.
    for x in range(len(restimulus)):

        start_gesture = restimulus[x][0][0].item()
        end_gesture = restimulus[x][0][-1].item()
        gesture = (start_gesture, end_gesture)

        if start_gesture == end_gesture:
            
            if non_transition_windows_left[gesture] > 0: 
                indices.append(x)
                non_transition_windows_left[gesture] -= 1
        
        else:
            if transition_windows_left[gesture] > 0:
                indices.append(x)
                transition_windows_left[gesture] -= 1

    return indices

def balance(restimulus, args):
    if args.transition_classifier:
        return balance_transition_classifier(restimulus, args)
    else:
        return balance_gesture_classifier(restimulus, args)


def contract(restim, args):
    if args.transition_classifier:
        return contract_transition_classifier(restim, args)
    else:
        return contract_gesture_classifier(restim, args)

def contract_gesture_classifier(restim, args):
    """Converts restimulus tensor to one-hot encoded tensor.

    Args:
        restim (tensor): restimulus data tensor
        unfold (bool, optional): whether data was unfolded according to time steps. Defaults to True.

    Returns:
        labels: restimulus data now one-hot encoded
    """
    numGestures = restim.max() + 1 # + 1 to account for rest gesture
    labels = torch.tensor(())


    labels = labels.new_zeros(size=(len(restim), numGestures))

    for x in range(len(restim)):
        if args.include_transitions:
            gesture = int(restim[x][0][-1]) # take the last gesture it belongs to (labels the transition as part of the gesture)
        else:
            gesture = int(restim[x][0][0])
        labels[x][gesture] = 1.0
    
    return labels

def contract_transition_classifier(restim, args):
    """Converts restimulus tensor to one-hot encoded tensor.

    Args:
        restim (tensor): restimulus data tensor
        unfold (bool, optional): whether data was unfolded according to time steps. Defaults to True.

    Returns:
        labels: restimulus data now one-hot encoded
    """
    transition_labels = torch.zeros((len(restim), 2), dtype=torch.float32)

    for x in range(len(restim)):

        start_gesture = restim[x][0][0].item()
        end_gesture = restim[x][0][-1].item()

        transition_labels[x] = torch.tensor([start_gesture, end_gesture], dtype=torch.float32)

    return transition_labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=200.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=200.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

def getRestim (n: int, exercise: int, unfold=True):
    """
    Returns a restiumulus (label) tensor for participant n and exercise exercise and if unfold, unfolded across time. 

    (Unfold=False is needed in getEMG for target normalization)

    Args:
        n (int): participant 
        exercise (int): exercise. 
        unfold (bool, optional): whether or not to unfold data across time steps. Defaults to True.
    """

    # read hdf5 file 
    restim = pd.read_hdf(f'DatasetsProcessed_hdf5/NinaproDB5/s{n}/restimulusS{n}_E{exercise}.hdf5')
    restim = torch.tensor(restim.values)

    # unfold extrcts sliding local blocks from a batched input tensor
    if (unfold):
        return restim.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
    return restim


def target_normalize (data, target_min, target_max, restim):

    assert data is not None, "Data is None"
    assert target_min is not None, "Target min is None"
    assert target_max is not None, "Target max is None"
    assert restim is not None, "Restim is None"

    source_min = np.zeros(numElectrodes, dtype=np.float32)
    source_max = np.zeros(numElectrodes, dtype=np.float32)

    resize = min(len(data), len(restim))
    data = data[:resize]
    restim = restim[:resize]
    
    for i in range(numElectrodes):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])

    data_norm = np.zeros(data.shape, dtype=np.float32)
    for gesture in range(target_min.shape[1]):
        if target_min[0][gesture] == 0 and target_max[0][gesture] == 0:
            continue
        for i in range(numElectrodes):
            data_norm[:, i] = data_norm[:, i] + (restim[:, 0] == gesture) * (((data[:, i] - source_min[i]) / (source_max[i] 
            - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture])
    return data_norm


def getEMG(input):
    """Returns EMG data for a given participant and exercise. EMG data is balanced (reduced rest gestures), target normalized (if toggled), filtered (butterworth), and unfolded across time. 

    Args:
        n (int): participant number
        exercise (int): exercise number
        target_min (np.array): minimum target values for each electrode
        target_max (np.array): maximum target values for each electrode
        leftout (int): participant number to leave out
        args: argument parser object (needed for DB3 to ignore subject 10, but can be ignored for DB5)

    Returns:
        (WINDOW, ELECTRODE, TIME STEP): EMG data
    """

    if (len(input) == 3):
        n, exercise, args = input
        leftout = None
        is_target_normalize = False
    else:
        n, exercise, target_min, target_max, leftout, args = input
        is_target_normalize = True

    emg = pd.read_hdf(f'DatasetsProcessed_hdf5/NinaproDB5/s{n}/emgS{n}_E{exercise}.hdf5')

    # emg = torch.tensor(emg.values) # to here 
    
    # normalize data for non leftout participants 
    if (is_target_normalize and n != leftout):
        np_data = np.array(emg.values) # target_normalize takes in np array
        R = np.array(getRestim(n, exercise, unfold=False))
        emg = torch.tensor(target_normalize(np_data, target_min, target_max, R))

    else:
        emg = torch.tensor(emg.values)

    restim = getRestim(n, exercise, unfold=True)
    return filter(emg.unfold(dimension=0, size=wLenTimesteps, step=stepLen)[balance(restim, args)])
    
def get_decrements(args):
    """
    Calculates how much gestures from exercise 1, 2, and 3 should be decremented by to make them sequential.

    Args:
        args: args parser object

    Returns:
        (d1, d2, d3): decrements for each exercise
    """
    
    decrements = {(1,): [0, 0, 0], (2,): [0, 17, 0], (3,): [0, 0, 40], (1,2): [0, 0, 0], (1,3): [0, 0, 23], (2,3): [0, 17, 17], (1,2,3): [0, 0, 0]}
    exercises = tuple(args.exercises)
    return decrements[exercises]

def make_gesture_sequential(gesture, args):
    """
    Removes missing gaps between gestures depending on which exercises are selected.

    Ex: If args.exercises = [1, 3], gesture labels in exercise 1 are kept the same while gesture labels in exercise 3 are decremented by 23. 

    Doing so prevents out of bound array accesses in train_test_split. 

    Returns:
        balanced_restim: restim but with gestures now sequential
    """
   
    exercise_starts = {1: 1, 2: 18, 3: 41}
    decrements = get_decrements(args)

    if gesture != 0: 
        exercise_group = (max(ex for ex in exercise_starts if exercise_starts[ex] <= gesture))-1
        d = decrements[exercise_group]
    else:
        d = 0

    return gesture - d
  
def getLabels (input):
    """Returns one-hot-encoding labels for a given participant and exercise. Labels are balanced (reduced rest gestures) and are sequential (no gaps between gestures of different exercises).

    Args:
        n (int): participant number
        exercise (int): exercise number
        args: argument parser object

    Returns:
        (TIME STEP, GESTURE): one-hot-encoded labels for participant n and exercise exercise
    """

    n, exercise, args = input
    restim = getRestim(n, exercise)             
    balanced_restim = restim[balance(restimulus=restim, args=args)]   # (WINDOW, GESTURE, TIME STEP) 
    return contract(restim=balanced_restim, args=args)

def getExtrema (n, proportion, exercise, args):
    
    """Returns the min max of the electrode per gesture for a proportion of its windows. 
    
    Used for target normalization.

    Args:
        n: participant
        proportion: proportion of windows to consider
        exercise: exercise
        args_exercises: exercises for the overall program (important for getLabels)

    Returns:
        (ELECTRODE, GESTURE): min and max values for each electrode per gesture

    """

    # Windowed data (must be windowed and balanced so that it matches the splitting in train_test_split)
    emg = getEMG((n, exercise, args))       # (WINDOW, ELECTRODE, TIME STEP)
    labels = getLabels((n, exercise, args))  # (TIME STEP, LABEL)

    # need to convert labels out of one-hot encoding
    num_gestures = labels.shape[1]
    labels = torch.argmax(labels, dim=1) 
    
    # Create new arrays to hold data
    mins = np.zeros((numElectrodes, num_gestures))   
    maxes = np.zeros((numElectrodes, num_gestures))

    # Get the proportion of the windows per gesture 
    unique_labels, counts = np.unique(labels, return_counts=True)
    size_per_gesture = np.round(proportion*counts).astype(int)
    gesture_amount = dict(zip(unique_labels, size_per_gesture)) # (GESTURE, NUMBER OF WINDOWS)

    for gesture in gesture_amount.keys():
        size_for_current_gesture = gesture_amount[gesture]

        all_windows = np.where(labels == gesture)[0]
        chosen_windows = all_windows[:size_for_current_gesture] 
        
        # out of these indices, pick the min/max emg values
        for j in range(numElectrodes): 
            # minimum emg value
            mins[j][gesture] = torch.min(emg[chosen_windows, j])
            maxes[j][gesture] = torch.max(emg[chosen_windows, j])

    return mins, maxes
           
              
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
    
    # Resize image
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image = resize(torch.from_numpy(image))
    
    # Get max and min values after interpolation
    max_val = image.max()
    min_val = image.min()
    
    # Contrast normalize again after interpolation
    image = (image - min_val) / (max_val - min_val)
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    
    return image.numpy().astype(np.float32)


def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def process_chunk(data_chunk):
    return np.apply_along_axis(calculate_rms, -1, data_chunk)

def process_optimized_makeOneImage(args_tuple):
    return optimized_makeOneImage(*args_tuple)

def process_optimized_makeOneMagnitudeImage(args_tuple):
    return optimized_makeOneMagnitudeImage(*args_tuple)

def process_optimized_makeOneImageChunk(args_tuple):
    images = [None] * len(args_tuple)
    for i in range(len(args_tuple)):
        images[i] = optimized_makeOneImage(*args_tuple[i])
    return images

def process_optimized_makeOneMagnitudeImageChunk(args_tuple):
    images = [None] * len(args_tuple)
    for i in range(len(args_tuple)):
        images[i] = optimized_makeOneMagnitudeImage(*args_tuple[i])
    return images

def closest_factors(num):
    # Find factors of the number
    factors = [(i, num // i) for i in range(1, int(np.sqrt(num)) + 1) if num % i == 0]
    # Sort factors by their difference, so the closest pair is first
    factors.sort(key=lambda x: abs(x[0] - x[1]))
    return factors[0]

def optimized_makeOneCWTImage(data, length, width, resize_length_factor, native_resnet_size):
    # Reshape and preprocess EMG data
    data = data.reshape(length, width).astype(np.float16)
    highest_cwt_scale = wLenTimesteps
    scales = np.arange(1, highest_cwt_scale)

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * highest_cwt_scale)
    width_to_transform_to = min(native_resnet_size, grid_length * width)

    time_frequency_emg = np.zeros((length * (highest_cwt_scale), width))

    # Perform Continuous Wavelet Transform (CWT)
    for i in range(length):
        frequencies, coefficients = fcwt.cwt(data[i, :], int(fs), int(scales[0]), int(scales[-1]), int(highest_cwt_scale))
        coefficients_abs = np.abs(coefficients) 
        # coefficients_dB = 10 * np.log10(coefficients_abs + 1e-12)  # Avoid log(0)
        time_frequency_emg[i * (highest_cwt_scale):(i + 1) * (highest_cwt_scale), :] = coefficients_abs

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(time_frequency_emg).float()
    emg_sample = emg_sample.view(numElectrodes, wLenTimesteps, -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, wLenTimesteps, -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    # Convert to RGB and resize
    data_converted = cmap(combined_image)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp and normalize
    image_clamped = torch.clamp(image_resized, 0, 1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Return final image as a NumPy array
    final_image = image_normalized.numpy().astype(np.float16)
    return final_image

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = wLenTimesteps // 4
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    Sxx_abs = np.abs(Sxx) # small constant added to avoid log(0)
    # Sxx_dB = 10 * np.log10(np.abs(Sxx_abs) + 1e-12)
    emg_sample = torch.from_numpy(Sxx_abs)
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])
    # flip spectrogram vertically for each electrode
    for i in range(numElectrodes):
        num_frequencies = len(frequencies)
        emg_sample[i*num_frequencies:(i+1)*num_frequencies, :] = torch.flip(emg_sample[i*num_frequencies:(i+1)*num_frequencies, :], dims=[0])

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(emg_sample).float()
    emg_sample = emg_sample.view(numElectrodes, len(frequencies), -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, len(frequencies), -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOnePhaseSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = wLenTimesteps // 4
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    
    # Sxx_abs = np.abs(Sxx) # small constant added to avoid log(0)

    Sxx_phase = np.angle(Sxx)
    Sxx_phase_normalized = (Sxx_phase + np.pi) / (2 * np.pi) 
    
    emg_sample = torch.from_numpy(Sxx_phase_normalized)
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])

    # flip spectrogram vertically for each electrode
    for i in range(numElectrodes):
        num_frequencies = len(frequencies)
        emg_sample[i*num_frequencies:(i+1)*num_frequencies, :] = torch.flip(emg_sample[i*num_frequencies:(i+1)*num_frequencies, :], dims=[0])

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(emg_sample).float()
    emg_sample = emg_sample.view(numElectrodes, len(frequencies), -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, len(frequencies), -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOneHilbertHuangImage(data, length, width, resize_length_factor, native_resnet_size):

    emg_sample = data 
    max_imfs = 6

    # Perform Empirical Mode Decomposition (EMD)
    intrinsic_mode_functions = emd.sift.sift(emg_sample, max_imfs=max_imfs-1) 
    instantaneous_phase, instantaneous_frequencies, instantaneous_amplitudes = \
        emd.spectra.frequency_transform(imf=intrinsic_mode_functions, sample_rate=fs, method='nht')
    
    # Pad any missing IMFs with zeros
    if instantaneous_phase.shape[-1] < max_imfs:
        padded_instantaneous_phase = np.zeros((instantaneous_phase.shape[0], max_imfs))

        for electrode_at_time in range(instantaneous_phase.shape[0]):
            missing_imfs = max_imfs - instantaneous_phase.shape[-1]
            padding = np.zeros(missing_imfs)
            padded_instantaneous_phase[electrode_at_time] = np.append(instantaneous_phase[electrode_at_time], padding)
        instantaneous_phase = padded_instantaneous_phase

    # Rearrange to be (WLENTIMESTEP, NUM_ELECTRODES, MAX_IMF+1 (includes a combined IMF))
    instantaneous_phase_norm = instantaneous_phase / (2 * np.pi) 
    emg_sample = np.array_split(instantaneous_phase_norm, numElectrodes, axis=0) 
    emg_sample = [torch.tensor(emg) for emg in emg_sample]
    emg_sample = torch.stack(emg_sample)
    emg_sample = emg_sample.permute(1, 0, 2) 

    # Stack the y axis to be all imfs per electrode
    final_emg = torch.zeros(wLenTimesteps, numElectrodes*(max_imfs))
    for t in range(wLenTimesteps):
        for i in range(numElectrodes):
            final_emg[t, i*(max_imfs):(i+1)*(max_imfs)] = emg_sample[t, i, :]

    combined_image = final_emg 
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()
    data_converted = cmap(data) 
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    length_to_transform_to = min(native_resnet_size, image.shape[-2])
    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_transform_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image
 

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, global_min=None, global_max=None,
              turn_on_spectrogram=False, turn_on_phase_spectrogram=False, turn_on_cwt=False, turn_on_hht=False):

    if standardScaler is not None:
        emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    else:
        emg = np.array(emg.view(len(emg), length*width))    # Use RMS preprocessing
        
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        # Apply RMS calculation along the last axis (axis=-1)
        emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = emg_rms  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)

    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        num_splits = multiprocessing.cpu_count() // 2
        data_chunks = np.array_split(emg, num_splits)
        
        emg_rms = process_map(process_chunk, data_chunks, chunksize=1, max_workers=num_splits, desc="Calculating RMS")
        # Apply RMS calculation along the last axis (axis=-1)
        # emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = np.concatenate(emg_rms)  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)
        
        del emg_rms
        del data_chunks

    # Parameters that don't change can be set once
    resize_length_factor = 1
    if turn_on_magnitude:
        resize_length_factor = 1
    native_resnet_size = 224

    args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
    chunk_size = len(args) // (multiprocessing.cpu_count() // 2)
    arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]
    images = []

    if not turn_on_magnitude and not turn_on_spectrogram and not turn_on_cwt and not turn_on_hht:
        for i in tqdm(range(len(arg_chunks)), desc="Creating Images in Chunks"):
            images.extend(process_optimized_makeOneImageChunk(arg_chunks[i]))

    if turn_on_magnitude:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        chunk_size = len(args) // (multiprocessing.cpu_count() // 2)
        arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]
        images_magnitude = []
        for i in tqdm(range(len(arg_chunks)), desc="Creating Magnitude Images in Chunks"):
            images_magnitude.extend(process_optimized_makeOneMagnitudeImageChunk(arg_chunks[i]))
        images = np.concatenate((images, images_magnitude), axis=2)

    elif turn_on_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Spectrogram Images"):
            images_spectrogram.append(optimized_makeOneSpectrogramImage(*args[i]))
        images = images_spectrogram

    elif turn_on_phase_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Phase Spectrogram Images"):
            images_spectrogram.append(optimized_makeOnePhaseSpectrogramImage(*args[i]))
        images = images_spectrogram

    elif turn_on_hht:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Phase HHT Images"):
            images_spectrogram.append(optimized_makeOneHilbertHuangImage(*args[i]))
        images = images_spectrogram
    
    elif turn_on_cwt:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_cwt_list = []
        # with multiprocessing.Pool(processes=5) as pool:
        for i in tqdm(range(len(emg)), desc="Creating CWT Images"):
            images_cwt_list.append(optimized_makeOneCWTImage(*args[i]))
        images = images_cwt_list
        
    elif turn_on_hht:
        raise NotImplementedError("HHT is not implemented yet")
    
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
    sn.set(font_scale=0.4)
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
    numGestures = len(gesture_labels)

    for i in range(numGestures):
        # Find indices
        gesture_indices = np.where(true_np == i)[0]

        # Select and denormalize only the required images
        gesture_images = denormalize(image_data[gesture_indices]).cpu().detach().numpy()
        average_images.append(np.mean(gesture_images, axis=0))

    average_images = np.array(average_images)

    # resize average images to 224 x 224
    resize = transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    average_images = np.array([resize(torch.from_numpy(img)).numpy() for img in average_images])

    # Plot average image of each gesture
    fig, axs = plt.subplots(2, 9, figsize=(15, 5))
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
    total_gestures = len(gesture_labels)  # Replace with the actual number of gestures

    # Create subplots
    fig, axs = plt.subplots(rows_per_gesture, total_gestures, figsize=(20, 15))

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


