#!/usr/bin/env python

import numpy as np
import scipy
import scipy.io as sio
from tqdm import tqdm
import pandas as pd
import h5py
import os
from glob import glob
import torch


# Argparse for including transitions and transition classifier
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
import argparse
parser = argparse.ArgumentParser(description="Include arguments for loading different data files")
parser.add_argument("--include_transitions", type=str2bool, help="Start labeling gestures as soon as cue begins. Defaults to False.", default=False)
parser.add_argument("--transition_classifier", type=str2bool, help="Classify windows as transtions or non transitions. Shortens observed window. Defaults to False.", default=False)
args = parser.parse_args()


# Sampling rate "Hz"
fs = 2000
gesture_names = ['Rest', 'Extension', 'Flexion', 'Ulnar_Deviation', 'Radial_Deviation', 'Grip', 'Abduction', 'Adduction', 'Supination', 'Pronation']

# Desired sEMG Segments Length in seconds
# Erase old files to write new hdf5 files if you want to change this


# Set time delay after cue
if args.include_transitions or args.transition_classifier:
    signal_segment_starting = 0 # start labeling as soon as the cue starts
else:
    signal_segment_starting = 1 

# Set second to stop observing after cue
if args.transition_classifier:
    signal_segment_ending = 2 # one second of each to balance transition/non-transition windows
else:
    signal_segment_ending = 6 # amount of time after signal. 6 seconds is the end of the cue, so 5 or 6 are good numbers


# Get sEMG Records directory
current_folder = './MCS_EMG/'  # Change with current folder
Base = os.path.join(current_folder, 'sEMG-dataset/raw/mat')  # Change raw or filtered as needed
Files = glob(os.path.join(Base, '**/*.mat'), recursive=True)
# Files.sort()  # Sort files if needed

                    
# Automated segmentation and saving of all participant sEMG data to sEMG gesture segment in HDF5
for file_path in Files:
    
    if args.transition_classifier:
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG_transition_classifier/'
    elif args.include_transitions:
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG_include_transitions/'
    else:
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG/'
    mat_data = scipy.io.loadmat(file_path)
    data = mat_data['data']
    participant_id = mat_data['iD'][0][0]  # Modify based on where ID is stored in your .mat

    # Create or open HDF5 file for each participant
    foldername = os.path.join(foldername, 'p' + str(participant_id) + '/')
    hdf5_filename = f'participant_{participant_id}.hdf5'
    hdf5_filename = os.path.join(foldername, hdf5_filename)
    # Make folders if they don't exist
    os.makedirs(os.path.dirname(hdf5_filename), exist_ok=True)
    # Open the file
    try:
        hdf_file = h5py.File(hdf5_filename, 'x')  # 'x' mode to fail if exists
    except FileExistsError:
        print(f"File {hdf5_filename} already exists. Skipping...")
        continue
    except OSError as e:
        print(e)
        print(f"Error creating file {hdf5_filename}. Skipping...")
        continue
    with h5py.File(hdf5_filename, 'a') as hdf_file:  # 'a' mode to append if already exists
        
        for rep in range(5):  # 5 repetitions
            rep_coeff = [4, 138, 272, 406, 540][rep]

            for gesture in range(10):  # Total of 10 hand gestures
                start_idx = (signal_segment_starting + rep_coeff + (gesture * 10)) * fs
                end_idx = ((rep_coeff + (gesture * 10)) + signal_segment_ending) * fs
                
                # Multi-channel sEMG data
                multi_channel_sEMG_data = data[start_idx:end_idx, :].T

                # Define a group for each gesture and cycle. Cycles act as hdf top-level keys 
                # and gestures act as sub-keys. 'sEMG' is the sole sub-sub-key for the data
                group_name = f'Cycle{rep + 1}/Gesture{gesture_names[gesture]}'
                grp = hdf_file.require_group(group_name)

                # Save the data in the group, formatted as (CHANNEL, TIME)
                grp.create_dataset('sEMG', data=multi_channel_sEMG_data, compression="gzip")
                
              

    print(f"Data for participant {participant_id} and all gestures saved in {hdf5_filename}.")


import h5py
import numpy as np
import os
import scipy.io  # For loading .mat files
from glob import glob
import matplotlib.pyplot as plt
#Flattened dataset with gestures as groups and numpy arrays of shape (CYCLE, CHANNEL, TIME)

for i in range(1, 41):

    if args.transition_classifier:
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG_transition_classifier/'
    elif args.include_transitions: 
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG_include_transitions/'
    else:
        foldername = 'DatasetsProcessed_hdf5/MCS_EMG/'

    foldername = os.path.join(foldername, 'p' + str(i) + '/')
    hdf5_input_filename = f'participant_{i}.hdf5'
    hdf5_input_filename = os.path.join(foldername, hdf5_input_filename)
    hdf5_output_filename = f'flattened_participant_{i}.hdf5'
    hdf5_output_filename = os.path.join(foldername, hdf5_output_filename)

    try: 
        hdf5_input_file = h5py.File(hdf5_input_filename, 'r')
    except FileNotFoundError:
        print(f"File {hdf5_input_filename} not found. Skipping...")
        continue
    # Open the existing HDF5 file to read
    try:
        hdf5_output_file = h5py.File(hdf5_output_filename, 'r')  
        print(f"File {hdf5_output_filename} exists.")
    except FileNotFoundError:
        print(f"File {hdf5_output_filename} not found. Reprocessing data...")

        with h5py.File(hdf5_input_filename, 'r') as hdf_read:
            try: 
                print(f"Attempting to create file {hdf5_output_filename}")
                # Open a new HDF5 file to write reprocessed data
                with h5py.File(hdf5_output_filename, 'w') as hdf_write:
                    
                    # Iterate through each gesture (assuming 10 gestures)
                    for gesture in range(10):
                        all_cycles_data = []  # List to hold data from all cycles for this gesture

                        # Collect data from each cycle for this gesture
                        for rep in range(1, 6):  # Assuming 5 cycles, named as 'Cycle1', 'Cycle2', ...
                            cycle_group_name = f'Cycle{rep}/Gesture{gesture_names[gesture]}'
                            if cycle_group_name in hdf_read:
                                # Read and append the data
                                cycle_data = hdf_read[cycle_group_name]['sEMG'][:]
                                all_cycles_data.append(cycle_data)

                        # Aggregate all cycles data into a single array (CYCLE, CHANNELS, TIME)
                        # hdf5 keys are named as 'Gesture{gesture_name}'    
                        if all_cycles_data:  # Ensure there is data to process
                            aggregated_data = np.stack(all_cycles_data, axis=0)  # Stack along new axis for cycles

                            # Write the aggregated data to the new file
                            hdf_write.create_dataset(f'Gesture{gesture_names[gesture]}', data=aggregated_data, compression="gzip")
            except FileExistsError as e:
                print(f"File {hdf5_output_filename} already exists. Skipping...")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue

        print(f"Reprocessed data saved in {hdf5_output_filename}.")


gesture_names = ['Rest', 'Extension', 'Flexion', 'Ulnar_Deviation', 'Radial_Deviation', 'Grip', 'Abduction', 'Adduction', 'Supination', 'Pronation']

if args.transition_classifier:
    foldername = 'DatasetsProcessed_hdf5/MCS_EMG_transition_classifier/p40/'
elif args.include_transitions:
    foldername = 'DatasetsProcessed_hdf5/MCS_EMG_include_transitions/p40/'
else:
    foldername = 'DatasetsProcessed_hdf5/MCS_EMG/p40/'

hdf5_filename = 'flattened_participant_40.hdf5'
hdf5_filename = os.path.join(foldername, hdf5_filename)
with h5py.File(hdf5_filename, 'r') as hdf_file:
    # List all groups
    print("Keys: %s" % hdf_file.keys())
    a_group_key = list(hdf_file.keys())[0]

    #print(hdf_file['GestureRest'].shape)
    for name in gesture_names:
        print(hdf_file["Gesture" + name].shape)


from scipy.signal import butter, filtfilt, iirnotch
gesture_names = ['Rest', 'Extension', 'Flexion', 'Ulnar_Deviation', 'Radial_Deviation', 'Grip', 'Abduction', 'Adduction', 'Supination', 'Pronation']
numGestures = 10
fs = 2000 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 100 #50 ms
def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=[5.0, 500.0], btype='bandpass', analog=False, fs=2000.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=2000.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

# Check flattened data
def getEMG (n):
    # NOTE: Not sure if this function is being used 

    if args.transition_classifier:
        file = h5py.File('DatasetsProcessed_hdf5/MCS_EMG_transition_classifier/p40/flattened_participant_40.hdf5', 'r')
    elif args.include_transitions:
        file = h5py.File('DatasetsProcessed_hdf5/MCS_EMG_include_transitions/p40/flattened_participant_40.hdf5', 'r')
    else:
        file = h5py.File('DatasetsProcessed_hdf5/MCS_EMG/p40/flattened_participant_40.hdf5', 'r')
    emg = []
    for gesture in gesture_names:
        data = filter(torch.from_numpy(np.array(file["Gesture" + gesture]))).unfold(dimension=-1, size=wLenTimesteps, step=stepLen)
        emg.append(torch.cat([data[i] for i in range(len(data))], dim=-2).permute((1, 0, 2)))
    return torch.cat(emg, dim=0)

# size of 4800 assumes 250 ms window
def getLabels (n):
    labels = np.zeros((4800, numGestures))
    for i in range(480):
        for j in range(numGestures):
            labels[j * 480 + i][j] = 1.0
    return labels

'''for i in range(40):
    data = getEMG(i)
    print(data.shape)'''

labels = getLabels(20)
numEach = [0 for i in range(10)]
for i in range(len(labels)):
    for j in range(len(labels[0])):
        if (labels[i][j] != 0):
            numEach[j] += 1

print(labels.shape)
print(numEach)

