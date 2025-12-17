import os
import pandas as pd
from pydub import AudioSegment
from itertools import product
import numpy as np
import random

# Define paths
folder_path = 'donateacry'
class_names = ['burping', 'discomfort', 'hungry', 'tired', 'belly_pain']
audio_files = {cls: [os.path.join(folder_path, cls, f) for f in os.listdir(os.path.join(folder_path, cls)) if f.endswith('.wav')] for cls in class_names}

# Maximum number of files to generate
max_files = 1000

def create_combinations(audio_files):
    # Create combinations of one file from each class
    all_combinations = list(product(*[files for files in audio_files.values()]))
    
    if len(all_combinations) > max_files:
        # Randomly sample max_files combinations
        sampled_combinations = random.sample(all_combinations, max_files)
    else:
        sampled_combinations = all_combinations
    
    return sampled_combinations

def concatenate_audio(files):
    combined = AudioSegment.empty()
    start_times = {}
    end_times = {}
    
    current_time = 0
    
    for file in files:
        cls=file.split('/')[1]
        audio = AudioSegment.from_wav(file)
        start_times[cls] = current_time / 1000.0
        combined += audio
        end_times[cls] = (current_time + len(audio)) / 1000.0
        current_time += len(audio)
    
    return combined, start_times, end_times

def save_combined_audio_and_txt(combinations, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    txt_data = []

    for idx, combo in enumerate(combinations):
        combined_audio, start_times, end_times = concatenate_audio(combo)

        # Save combined audio
        audio_filename = os.path.join(output_folder, f'combined_audio_{idx}.wav')
        combined_audio.export(audio_filename, format='wav')

        # Prepare text data
        for cls in start_times.keys():
            # Prepare a row with values separated by tabs
            row = f"{audio_filename}\tstreet\t{start_times[cls]}\t{end_times[cls]}\t{cls}"
            txt_data.append(row)

    # Save TXT file without column headers
    txt_filename = os.path.join(output_folder, 'audio_metadata.txt')
    with open(txt_filename, 'w') as txt_file:
        txt_file.write("\n".join(txt_data))

# Main execution
combinations = create_combinations(audio_files)
shuffled_combinations=[]
for tup in combinations:
    lst = list(tup)  # Convert tuple to list
    random.shuffle(lst)  # Shuffle the list
    shuffled_combinations.append(tuple(lst))

output_folder = 'sed_cry/audio/street'
save_combined_audio_and_txt(shuffled_combinations, output_folder)

print("Processing complete. Check the f'{output_folder}' folder for results.")

