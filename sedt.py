import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plot
import utils  # Ensure utils.py contains the required helper functions
import time
import metrics  # Ensure metrics.py contains the compute_scores function
import math

# Define class labels
__class_labels = {
    'hungry': 0,
    'discomfort': 1,
    'tired': 2,
    'belly_pain': 3,
    'burping': 4
}

# Define the Transformer model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=8, hidden_dim=192, dropout=0.3):
        super(TransformerModel, self).__init__()

        # Multi-scale CNN Feature Extractor
        self.cnn1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)

        # Positional Encoding
        self.embedding = nn.Linear(hidden_dim // 2, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated Linear Unit
        self.glu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GLU()
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, nb_ch, seq_len, input_dim = x.shape

        x = torch.mean(x, dim=1)  # Combine channels -> (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len) for CNN

        # Multi-scale CNN feature extraction
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.bn1(x)

        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, hidden_dim//2)

        # Embedding & Positional Encoding
        x = self.embedding(x)
        x = self.norm(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Gated Linear Unit
        x = self.glu(x)

        # Final Prediction
        x = self.fc(x)  # (batch_size, seq_len, num_classes)
        return x


def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test

# Plot training and validation metrics
def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f1')
    plot.plot(range(_nb_epoch), _er, label='error rate')
    plot.legend()
    plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()

# Preprocess data into sequences and multi-channel format
def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # Split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    # Split into multi-channel format
    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test

# Define a linear learning rate scheduler
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Main script
if __name__ == '__main__':
    start = time.time()

    is_mono = True  # True: mono-channel input, False: binaural input
    feat_folder = './feat/'
    __fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

    nb_ch = 1 if is_mono else 2
    batch_size = 128    # Decrease this if you want to run on smaller GPU's
    seq_len = 256       # Frame sequence length. Input to the CRNN.
    nb_epoch = 500      # Training epochs
    patience = int(0.25 * nb_epoch)  # Patience for early stopping

    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    sr = 8000
    nfft = 512
    frames_1_sec = int(sr / (nfft / 2.0))

    print('\n\nUNIQUE ID: {}'.format(__fig_name))
    print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
        nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

    # Folder for saving model and training curves
    __models_dir = 'models/'
    utils.create_folder(__models_dir)

    # Initialize lists to store metrics for all folds
    avg_er, avg_f1, avg_acc, avg_prec, avg_rec, avg_spec, avg_logloss, avg_auc_roc = [], [], [], [], [], [], [], []

    for fold in [1, 2, 3, 4]:
        print('\n\n----------------------------------------------')
        print('FOLD: {}'.format(fold))
        print('----------------------------------------------\n')

        # Load feature and labels, pre-process it
        X, Y, X_test, Y_test = load_data(feat_folder, is_mono, fold)
        X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

        print("Shape of X:", X.shape)
        print("Shape of Y:", Y.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of Y_test:", Y_test.shape)

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = TransformerModel(input_dim=X.shape[3], num_classes=len(__class_labels))
        print(summary(model))
        optimizer = AdamW(model.parameters(), lr=0.001)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * nb_epoch)
        criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss for multi-label classification

        # Track metrics
        best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, float('inf'), 0, None
        tr_loss, val_loss = [0] * nb_epoch, [0] * nb_epoch
        f1_overall_1sec_list, er_overall_1sec_list, acc_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
        prec_overall_1sec_list, rec_overall_1sec_list, spec_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
        logloss_overall_1sec_list, auc_roc_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch

        for i in range(nb_epoch):
            print('Epoch : {} '.format(i), end='')
            model.train()
            epoch_tr_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)  # Shape: (batch_size, seq_len, num_classes)

                # Reshape outputs and labels for BCEWithLogitsLoss
                outputs = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * seq_len, num_classes)
                labels = labels.view(-1, labels.size(-1))    # Shape: (batch_size * seq_len, num_classes)

                # Compute loss
                loss = criterion(outputs, labels)  # BCEWithLogitsLoss expects float labels
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_tr_loss += loss.item()

            tr_loss[i] = epoch_tr_loss / len(train_loader)

            # Evaluate on test set
            model.eval()
            epoch_val_loss = 0  # Track validation loss
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch
                    outputs = model(inputs)  # Shape: (batch_size, seq_len, num_classes)

                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)  # Shape: (batch_size, seq_len, num_classes)

                    # Compute validation loss
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1, labels.size(-1))
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()

                    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
                    preds = (probs > 0.5).int()  # Shape: (batch_size, seq_len, num_classes)

                    # Reshape and collect predictions and labels
                    y_true.extend(labels.view(-1, labels.size(-1)).cpu().numpy())  # Shape: (batch_size * seq_len, num_classes)
                    y_pred.extend(preds.view(-1, preds.size(-1)).cpu().numpy())    # Shape: (batch_size * seq_len, num_classes)

            val_loss[i] = epoch_val_loss / len(test_loader)  # Append average validation loss

            # Calculate metrics using the modular function
            score_list = metrics.compute_scores(np.array(y_pred), np.array(y_true), frames_in_1_sec=frames_1_sec)

            # Extract metrics from score_list
            f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
            er_overall_1sec_list[i] = score_list['er_overall_1sec']
            acc_overall_1sec_list[i] = score_list['accuracy']
            prec_overall_1sec_list[i] = score_list['precision']
            rec_overall_1sec_list[i] = score_list['recall']
            spec_overall_1sec_list[i] = score_list['specificity']
            logloss_overall_1sec_list[i] = score_list['log_loss']
            auc_roc_overall_1sec_list[i] = score_list['auc_roc']

            # Calculate confusion matrix
            test_pred_cnt = np.sum(np.array(y_pred), axis=1)
            Y_test_cnt = np.sum(np.array(y_true), axis=1)
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
                torch.save(model.state_dict(), os.path.join(__models_dir, f'{__fig_name}_fold_{fold}_best_model.pth'))
                best_epoch = i
                pat_cnt = 0

            print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {}, ACC_overall : {}, PREC_overall : {}, REC_overall : {}, SPEC_overall : {}, LOGLOSS_overall : {}, AUCROC_overall : {}, Best ER : {}, best_epoch: {}'.format(
                    tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i], acc_overall_1sec_list[i], prec_overall_1sec_list[i], rec_overall_1sec_list[i], spec_overall_1sec_list[i], logloss_overall_1sec_list[i], auc_roc_overall_1sec_list[i], best_er, best_epoch))
            plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_fold_{}'.format(fold))
            if pat_cnt > patience:
                break

        # Store metrics for this fold
        avg_er.append(best_er)
        avg_f1.append(f1_for_best_er)
        avg_acc.append(acc_for_best_er)
        avg_prec.append(prec_for_best_er)
        avg_rec.append(rec_for_best_er)
        avg_spec.append(spec_for_best_er)
        avg_logloss.append(logloss_for_best_er)
        avg_auc_roc.append(auc_roc_for_best_er)

        print('saved model for the best_epoch: {} with best_er: {} f1_for_best_er: {} acc_for_best_er: {} prec_for_best_er: {} rec_for_best_er: {} spec_for_best_er: {} logloss_for_best_er: {} auc_roc_for_best_er: {}'.format(
            best_epoch, best_er, f1_for_best_er, acc_for_best_er, prec_for_best_er, rec_for_best_er, spec_for_best_er, logloss_for_best_er, auc_roc_for_best_er))
        print('best_conf_mat: {}'.format(best_conf_mat))
        print('best_conf_mat_diag: {}'.format(np.diag(best_conf_mat)))

    print('\n\nMETRICS FOR ALL FOUR FOLDS: avg_er: {}, avg_f1: {}, avg_acc: {}, avg_prec: {}, avg_rec: {}, avg_spec: {}, avg_logloss: {}, avg_auc_roc: {}'.format(avg_er, avg_f1, avg_acc, avg_prec, avg_rec, avg_spec, avg_logloss, avg_auc_roc))
    print('MODEL AVERAGE OVER FOUR FOLDS: avg_er: {}, avg_f1: {}, avg_acc: {}, avg_prec: {}, avg_rec: {}, avg_spec: {}, avg_logloss: {}, avg_auc_roc: {}'.format(np.mean(avg_er), np.mean(avg_f1), np.mean(avg_acc), np.mean(avg_prec), np.mean(avg_rec), np.mean(avg_spec), np.mean(avg_logloss), np.mean(avg_auc_roc)))

    end = time.time()
    total_time = end - start
    print("\ntotal time this script took: "+ str(total_time))



