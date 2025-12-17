import wave
import numpy as np
import utils
import librosa
from sklearn import preprocessing
import os
import torch
import torchaudio
import torchaudio.transforms as T

# ---------------- HuBERT Setup ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hubert_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
hubert_model.eval()

def extract_hubert(_y, _sr):
    """
    Extract pretrained HuBERT embeddings from the audio signal.

    Parameters
    ----------
    _y : numpy.ndarray
        Audio signal (waveform)
    _sr : int
        Sample rate of the audio signal

    Returns
    -------
    embeddings : numpy.ndarray
        HuBERT features of shape (n_time_frames, n_features)
    """
    waveform = torch.tensor(_y, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Resample to 16 kHz if needed
    if _sr != 16000:
        resampler = T.Resample(_sr, 16000)
        waveform = resampler(waveform)
    
    with torch.inference_mode():
        embeddings, _ = hubert_model(waveform)
        # embeddings: [1, n_frames, n_features]
    
    return embeddings.squeeze(0).cpu().numpy()


# ---------------- Audio Loading ----------------
def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array, supports 24-bit wav-format"""
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('Data length not multiple of sample size * channels')
        if sample_width > 4:
            raise ValueError('Sample width > 4 bytes not supported')

        if sample_width == 3:
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.frombuffer(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            audio_data = np.mean(audio_data, axis=0)

        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

# ---------------- Label Loading ----------------
def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict

# ---------------- Main Script ----------------
is_mono = True
__class_labels = {
    'hungry': 0,
    'discomfort': 1,
    'tired': 2,
    'belly_pain': 3,
    'burping': 4
}

folds_list = [1, 2, 3, 4]
evaluation_setup_folder = '../../combined_data_300/meta/evaluation_setup'
audio_folder = '../../combined_data_300/audio/street'

feat_folder = './feat_hubert/'  # Feature folder for HuBERT outputs
utils.create_folder(feat_folder)

sr = 16000  # HuBERT expects 16 kHz
hop_len_hubert = 320  # approx 20ms per frame for 16kHz HuBERT

# Load labels
train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(1))
evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
desc_dict = load_desc_file(train_file)
desc_dict.update(load_desc_file(evaluate_file))

# Extract features for all audio files, and save with labels
for audio_filename in os.listdir(audio_folder):
    audio_file = os.path.join(audio_folder, audio_filename)
    print(f'Extracting HuBERT features and label for: {audio_file}')

    y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
    embeddings = None

    if is_mono:
        embeddings = extract_hubert(y, sr)
    else:
        for ch in range(y.shape[0]):
            emb_ch = extract_hubert(y[ch, :], sr)
            if embeddings is None:
                embeddings = emb_ch
            else:
                embeddings = np.concatenate((embeddings, emb_ch), axis=1)

    # Create label matrix aligned to HuBERT frames
    label = np.zeros((embeddings.shape[0], len(__class_labels)))
    tmp_data = np.array(desc_dict[audio_filename])
    frame_start = np.floor(tmp_data[:, 0] * sr / hop_len_hubert).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len_hubert).astype(int)
    se_class = tmp_data[:, 2].astype(int)

    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = 1

    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
    np.savez(tmp_feat_file, embeddings, label)

# ---------------- Feature Normalization ----------------
for fold in folds_list:
    train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(fold))
    evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(fold))
    train_dict = load_desc_file(train_file)
    test_dict = load_desc_file(evaluate_file)

    X_train, Y_train, X_test, Y_test = None, None, None, None

    for key in train_dict.keys():
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
        dmp = np.load(tmp_feat_file)
        tmp_emb, tmp_label = dmp['arr_0'], dmp['arr_1']
        if X_train is None:
            X_train, Y_train = tmp_emb, tmp_label
        else:
            X_train, Y_train = np.concatenate((X_train, tmp_emb), 0), np.concatenate((Y_train, tmp_label), 0)

    for key in test_dict.keys():
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
        dmp = np.load(tmp_feat_file)
        tmp_emb, tmp_label = dmp['arr_0'], dmp['arr_1']
        if X_test is None:
            X_test, Y_test = tmp_emb, tmp_label
        else:
            X_test, Y_test = np.concatenate((X_test, tmp_emb), 0), np.concatenate((Y_test, tmp_label), 0)

    # Normalize features
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    normalized_feat_file = os.path.join(feat_folder, 'hubert_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
    print(f'normalized_feat_file : {normalized_feat_file}')

