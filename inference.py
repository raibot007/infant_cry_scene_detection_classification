import os
import numpy as np
from keras.models import load_model
import utils
from sklearn import preprocessing
from pydub import AudioSegment
import wave
import librosa
from kymatio import Scattering1D  
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import hashlib


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format

    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

# Define WST extraction function
def extract_wst(_y, _sr, J=8, Q=1):
    """
    Extract Wavelet Scattering Transform (WST) features from the audio signal.

    Parameters
    ----------
    _y : numpy.ndarray
        Audio signal (waveform)

    _sr : int
        Sample rate of the audio signal

    J : int
        Log-scale parameter controlling the time scale of wavelets. Default is 8.

    Q : int
        Number of wavelets per octave. Default is 1.

    Returns
    -------
    wst : numpy.ndarray
        Scattering coefficients (features) of shape (n_time_frames, n_wst_features)
    """
    # Create Scattering1D object for WST extraction
    T = len(_y)  # length of the signal
    scattering = Scattering1D(J=J, shape=(T,), Q=Q, max_order=2)

    # Apply WST to the audio signal
    Sx = scattering(_y)

    # Sx is of shape (n_wavelets, n_time_frames), convert it to time-major
    return Sx.T  # Return features with shape (n_time_frames, n_wst_features)


input="2_d_female_hypoglycemia_001.ogg"
audio = AudioSegment.from_ogg(input)
audio.export("6_m_male.wav", format="wav")
test_audio="6_m_male.wav"


model_path = './models/mon_2025_03_06_10_53_27_fold_3_model.h5'
seq_len = 128
nb_ch = 1
sr = 8000
J = 7
Q = 12
hop_len_wst = 2 ** J
posterior_thresh = 0.5

audio, sr = load_audio(test_audio, mono=True, fs=sr)

features = extract_wst(audio, sr, J=J, Q=Q)

train_data = np.load('./feat_wst/wst_mon_fold3.npz')
scaler = preprocessing.StandardScaler()
scaler.fit(train_data['arr_0'])  
features_norm = scaler.transform(features)


features_seq = utils.split_in_seqs(features_norm, seq_len)  # (num_seq, seq_len, feat_dim)
features_seq = utils.split_multi_channels(features_seq, nb_ch)

model = load_model(model_path)

predictions = model.predict(features_seq)  # (num_seq, seq_len, num_classes)
predictions_bin = predictions > posterior_thresh
print(predictions_bin)
frame_preds = predictions_bin.reshape(-1, predictions.shape[-1])  # shape: (total_frames, num_classes)
frame_duration = hop_len_wst / sr  # duration of each frame in seconds

segments = []
start_idx = 0
prev_active = set(np.where(frame_preds[0] == 1)[0])

for i in range(1, len(frame_preds)):
    current_active = set(np.where(frame_preds[i] == 1)[0])
    
    if current_active != prev_active:
        if prev_active:
            onset = round(start_idx * frame_duration, 2)
            offset = round(i * frame_duration, 2)
            segments.append({
                'onset': onset,
                'offset': offset,
                'classes': sorted(list(prev_active))
            })
        start_idx = i
        prev_active = current_active

# Handle last segment
if prev_active:
    onset = round(start_idx * frame_duration, 2)
    offset = round(len(frame_preds) * frame_duration, 2)
    segments.append({
        'onset': onset,
        'offset': offset,
        'classes': sorted(list(prev_active))
    })

# Create DataFrame
df_segments = pd.DataFrame(segments)
print(df_segments)


class_map = {
    0: "hungry",
    1: "discomfort",
    2: "tired",
    3: "belly pain",
    4: "burping"
}

# Load audio for waveform visualization
waveform, _ = librosa.load(test_audio, sr=sr)
time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))

plt.figure(figsize=(15, 6))

# Plot the waveform
plt.plot(time_axis, waveform, label="Audio waveform", color='black', linewidth=1)


# Build a unique color for each class combination using hashing
def get_color_for_classes(cls_tuple):
    # Generate a deterministic color from the class combination tuple
    hash_val = int(hashlib.md5(str(cls_tuple).encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2 ** 32))
    return np.random.rand(3,)  # RGB color

# Plot segments with colored background for each class
colors = plt.cm.get_cmap('tab10')

# Track which labels are already plotted for legend de-duplication
plotted_labels = set()

for _, row in df_segments.iterrows():
    onset = row['onset']
    offset = row['offset']
    active_classes = tuple(sorted(row['classes']))
    if not active_classes:
        continue
    label = ', '.join([class_map.get(c, str(c)) for c in active_classes])
    color = get_color_for_classes(active_classes)

    if label not in plotted_labels:
        plt.axvspan(onset, offset, color=color, alpha=0.3, label=label)
        plotted_labels.add(label)
    else:
        plt.axvspan(onset, offset, color=color, alpha=0.3)

plt.title("Audio waveform with predicted cry segments")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

# Legend
plt.legend(loc="upper right")

# Save the figure
output_path = "prediction_plot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")

plt.close()

