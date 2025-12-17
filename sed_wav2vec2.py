from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Permute
from keras.layers import BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import metrics
import utils
from IPython import embed
import keras.backend as K
K.set_image_data_format('channels_last')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)

#######################################################################################
# UPDATED: load_data now loads Wav2Vec2 features instead of HuBERT
#######################################################################################

def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'wav2vec2_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test


#######################################################################################
# UPDATED: model renamed for Wav2Vec2 embeddings (but same logic)
#######################################################################################

def get_wav2vec2_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):
    print("Input shape:", data_in.shape)

    # Input for Wav2Vec2 embeddings: (time_frames, embedding_dim, channels)
    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start

    # CNN Layers for feature extraction
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    # Permuting dimensions to align for RNN input
    spec_x = Permute((2, 1, 3))(spec_x)
    print("Shape after Permute:", spec_x.shape)

    # Reshaping for RNN
    spec_x = Reshape((data_in.shape[-2], -1))(spec_x)
    print("Shape after Reshape:", spec_x.shape)

    # RNN layers
    for _r in _rnn_nb:
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul')(spec_x)

    # Fully connected layers
    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    # Output layer
    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy')
    _model.summary()
    return _model


#######################################################################################
# Helper plotting & preprocessing
#######################################################################################

def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()
    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend(); plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend(); plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)
    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)
    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

start = time.time()

is_mono = True  # True: mono-channel input, False: binaural input

feat_folder = './feat_wav2vec2/'   # UPDATED folder
__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

nb_ch = 1 if is_mono else 2
batch_size = 128
seq_len = 128
nb_epoch = 500
patience = int(0.25 * nb_epoch)

# Wav2Vec2 frame rate ≈ 20ms → 50 frames/sec
frames_1_sec = 50

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving models and plots
__models_dir = 'models_wav2vec2/'
utils.create_folder(__models_dir)

# Model parameters
cnn_nb_filt = 128
cnn_pool_size = [4, 2]
rnn_nb = [32, 32]
fc_nb = [32]
dropout_rate = 0.3
print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

avg_er, avg_f1, avg_acc, avg_prec, avg_rec, avg_spec, avg_logloss, avg_auc_roc = ([] for _ in range(8))

for fold in [1, 2, 3, 4]:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')

    # Load Wav2Vec2 embeddings and labels
    X, Y, X_test, Y_test = load_data(feat_folder, is_mono, fold)
    X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of Y_test:", Y_test.shape)

    # Build model
    model = get_wav2vec2_model(X, Y, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb)

    # Training loop
    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0]*nb_epoch, [0]*nb_epoch, [0]*nb_epoch, [0]*nb_epoch
    acc_overall_1sec_list, prec_overall_1sec_list, rec_overall_1sec_list = [0]*nb_epoch, [0]*nb_epoch, [0]*nb_epoch
    spec_overall_1sec_list, logloss_overall_1sec_list, auc_roc_overall_1sec_list = [0]*nb_epoch, [0]*nb_epoch, [0]*nb_epoch
    posterior_thresh = 0.5

    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')
        hist = model.fit(
            X, Y,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            epochs=1,
            verbose=2
        )
        val_loss[i] = hist.history.get('val_loss')[-1]
        tr_loss[i] = hist.history.get('loss')[-1]

        pred = model.predict(X_test)
        pred_thresh = pred > posterior_thresh
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        acc_overall_1sec_list[i] = score_list['accuracy']
        prec_overall_1sec_list[i] = score_list['precision']
        rec_overall_1sec_list[i] = score_list['recall']
        spec_overall_1sec_list[i] = score_list['specificity']
        logloss_overall_1sec_list[i] = score_list['log_loss']
        auc_roc_overall_1sec_list[i] = score_list['auc_roc']
        pat_cnt += 1

        test_pred_cnt = np.sum(pred_thresh, 2)
        Y_test_cnt = np.sum(Y_test, 2)
        conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
        conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

        if er_overall_1sec_list[i] < best_er:
            best_conf_mat = conf_mat
            best_er = er_overall_1sec_list[i]
            f1_for_best_er = f1_overall_1sec_list[i]
            acc_for_best_er = acc_overall_1sec_list[i]
            prec_for_best_er = prec_overall_1sec_list[i]
            rec_for_best_er = rec_overall_1sec_list[i]
            spec_for_best_er = spec_overall_1sec_list[i]
            logloss_for_best_er = logloss_overall_1sec_list[i]
            auc_roc_for_best_er = auc_roc_overall_1sec_list[i]
            model.save(os.path.join(__models_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
            best_epoch = i
            pat_cnt = 0

        print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {}, ACC : {}, PREC : {}, REC : {}, SPEC : {}, LOGLOSS : {}, AUCROC : {}, Best ER : {}, best_epoch: {}'.format(
            tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i],
            acc_overall_1sec_list[i], prec_overall_1sec_list[i], rec_overall_1sec_list[i],
            spec_overall_1sec_list[i], logloss_overall_1sec_list[i], auc_roc_overall_1sec_list[i],
            best_er, best_epoch))

        plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_fold_{}'.format(fold))
        if pat_cnt > patience:
            break

    avg_er.append(best_er)
    avg_f1.append(f1_for_best_er)
    avg_acc.append(acc_for_best_er)
    avg_prec.append(prec_for_best_er)
    avg_rec.append(rec_for_best_er)
    avg_spec.append(spec_for_best_er)
    avg_logloss.append(logloss_for_best_er)
    avg_auc_roc.append(auc_roc_for_best_er)

    print('saved model for best_epoch: {} with best_er: {} f1_for_best_er: {}'.format(best_epoch, best_er, f1_for_best_er))
    print('best_conf_mat: {}'.format(best_conf_mat))
    print('best_conf_mat_diag: {}'.format(np.diag(best_conf_mat)))

print('\n\nMETRICS FOR ALL FOUR FOLDS:')
print('avg_er: {}, avg_f1: {}, avg_acc: {}, avg_prec: {}, avg_rec: {}, avg_spec: {}, avg_logloss: {}, avg_auc_roc: {}'.format(
    avg_er, avg_f1, avg_acc, avg_prec, avg_rec, avg_spec, avg_logloss, avg_auc_roc))
print('MODEL AVERAGE OVER FOUR FOLDS:')
print('avg_er: {}, avg_f1: {}, avg_acc: {}, avg_prec: {}, avg_rec: {}, avg_spec: {}, avg_logloss: {}, avg_auc_roc: {}'.format(
    np.mean(avg_er), np.mean(avg_f1), np.mean(avg_acc), np.mean(avg_prec),
    np.mean(avg_rec), np.mean(avg_spec), np.mean(avg_logloss), np.mean(avg_auc_roc)))

end = time.time()
print("\nTotal time this script took: " + str(end - start))

