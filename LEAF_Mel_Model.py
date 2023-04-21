from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from torch.utils.data import DataLoader
import torch
from torchaudio import transforms
import torchaudio
from torch_audiomentations import AddColoredNoise, ApplyImpulseResponse
from torch import nn
from torch.nn import init
import os
import glob
import ntpath
import numpy as np
from leaf_pytorch.frontend import Leaf
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sn
import pandas as pd
import warnings
from pathlib import Path
import random

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

# Model adapted from:
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

# leaf_pytorch: commented out print(output.shape) in filters.py file to supress output shape printout

# specifications for operation
frontend = "LEAF"                       # define frontend ("MEL", "LEAF", "leafPCEN", "leafFB")
train_model = True                      # activate training
test_model = True                       # activate testing
test_point = 57                         # for testing trained model
early_stop = True                       # activate early stopping
augmentation = True                     # activate data augmentation
dataset = 47                            # specify dataset to use (32, 47, 66)
layers = 4                              # specify number of convolutional layers to use (4 or 5)
seed = 1000                             # set seed to control randomization

# parameters
weight_decay = 0.001                    # set weight decay
dropout = 0.23                          # set dropout rate
max_patience = 8                        # define ES patience
n_epochs = 1                            # number of epochs for training when ES is disabled
N_CHANNELS = 1                          # number of input channels
SAMPLE_RATE = 44100                     # sample rate of input files
audio_len = 5                           # length of audio files
HOP_LENGTH = int((SAMPLE_RATE*5)/1500)  # hop length for Mel
leaf_hop = 3.335                        # equivalent hop length for LEAF
BATCH_SIZE = 14                         # batch size for training and evaluation
N_MELS = 64                             # number of filters for Mel and LEAF
top_db = 80                             # set max dB for decibel conversion

# determinism
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# high number of epochs when ES is enabled
if early_stop:
    num_epochs = 1000
else:
    num_epochs = n_epochs

train_data_path = f"{os.getcwd()}/Datasets/{dataset} Train Chunks/"               # training dataset
validation_data_path = f"{os.getcwd()}/Datasets/{dataset} Validation Chunks/"     # validation dataset
test_data_path = f"{os.getcwd()}/Datasets/{dataset} Test Chunks/"                 # test dataset

Path(os.getcwd() + f"/Results/").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {frontend} on {device}")

# create list of species and assign class identifier
ID_list = []

for filepath in glob.glob(os.path.join(test_data_path, '*.wav')):
    file_ID = ntpath.basename(filepath)                                 # filename without path
    species_ID = file_ID.split("_")[0]                                  # only get characters containing the species

    if species_ID not in ID_list:
        ID_list.append(species_ID)

ID_list = sorted(ID_list)
ID_list = dict.fromkeys(ID_list)

number = 0
for ID_species in ID_list:
    ID_list[ID_species] = number
    number += 1

if len(ID_list) != dataset:
    print("Number of species does not match dataset")

# get list of species from sorted csv file
classes_dict = pd.read_csv(f"{os.getcwd()}/Datasets/{dataset}_species_order.csv", delimiter="\t", header=None)
classes_dict = classes_dict[0].tolist()
classes_dict = dict.fromkeys(classes_dict)

for class_ID in classes_dict:
    classes_dict[class_ID] = ID_list[class_ID]

classes_list = list(classes_dict.values())


# creates dataframes with file path and class ID for all files in the three datasets
def annotate(data_path):
    species_list = []  # init species list
    files_list = []  # init files list
    class_ID = []  # init list of classIDs per species

    for filename in glob.glob(os.path.join(data_path, '*.wav')):
        file = ntpath.basename(filename)        # filename without path
        species = file.split("_")[0]            # only get characters containing the species
        ID = dict.get(ID_list, str(species))    # get species ID
        class_ID.append(ID)                      # get classIDs for each file
        species_list.append(species)            # append species and files to lists
        files_list.append(file)

    # combine lists, sort and create dataframe for data loader
    metadata_file = list(zip(class_ID, species_list, files_list))
    metadata_file.sort()
    col_names = ["classID", "species_name", "file_name"]
    df = pd.DataFrame(metadata_file, columns=col_names)
    df["path_name"] = data_path + df["file_name"]
    df = df[["path_name", "classID"]]
    return df


# open files, truncate number of samples, preprocess for frontends
class AudioUtil:
    @staticmethod
    def open(audio_file):                                           # open files
        sig, sr = torchaudio.load(audio_file)

        if sig.size(dim=1) < sr * audio_len:                        # pad signal if too short
            pad_value = (sig.size(dim=1) - sr * audio_len) * (-1)
            sig = F.pad(sig, (0, pad_value))

        if sig.size(dim=1) > sr * audio_len:                        # truncate signal if too long
            sig = sig.numpy()
            sig = sig[:sr * audio_len]
            sig = torch.tensor(sig)

        return sig, sr


# define datasets
class SoundDS:
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.sr = SAMPLE_RATE
        self.channel = N_CHANNELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, "path_name"]      # Absolute file path of the audio file
        class_id = self.df.loc[idx, 'classID']          # Get the Class ID
        aud, sr = AudioUtil.open(audio_file)            # open file

        return aud, class_id


# create dataframes for datasets
train_df = annotate(train_data_path)
validation_df = annotate(validation_data_path)
test_df = annotate(test_data_path)

# create datasets
train_dataset = SoundDS(train_df, train_data_path)
validation_dataset = SoundDS(validation_df, validation_data_path)
test_dataset = SoundDS(test_df, test_data_path)

num_train = len(train_dataset)
num_val = len(validation_dataset)
num_test = len(test_dataset)

notes = f"weight decay = {weight_decay}, dropout = {dropout} and {layers} layers"

if augmentation:
    print(f"{num_train} files and {len(ID_list)} taxa on seed {seed}, {notes}")
    augment = "aug"         # for file names
    augment_train = True    # activate augmentation in training dataset
else:
    print(f"{num_train} files and {len(ID_list)} taxa on seed {seed} without augmentation, {notes}")
    augment = "no_aug"      # for file names
    augment_train = False   # deactivate augmentation in training dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

# Create training, validation and test data loaders
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       worker_init_fn=seed_worker, generator=g)

validation_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            worker_init_fn=seed_worker, generator=g)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      worker_init_fn=seed_worker, generator=g)


# define model
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # torch-audiomentations
        self.noise = AddColoredNoise(p=0.9, p_mode="per_example", mode="per_example", sample_rate=SAMPLE_RATE,
                                     min_snr_in_db=25, max_snr_in_db=40, min_f_decay=-2, max_f_decay=1.5)

        self.impulse = ApplyImpulseResponse(p=0.7, p_mode="per_example", sample_rate=SAMPLE_RATE, mode="per_example",
                                            compensate_for_propagation_delay=True, ir_paths="44100 IRs")

        # Mel frontend
        self.spectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                                                     n_fft=1000, f_max=SAMPLE_RATE / 2, win_length=int(HOP_LENGTH * 2))

        self.amp_db = transforms.AmplitudeToDB(top_db=top_db)

        # LEAF frontend
        self.leaf = Leaf(n_filters=N_MELS, sample_rate=SAMPLE_RATE, window_stride=leaf_hop, window_len=leaf_hop*2,
                         init_max_freq=SAMPLE_RATE/2)

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(N_CHANNELS, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(N_CHANNELS*8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(N_CHANNELS*16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(N_CHANNELS*32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        if layers == 5:
            self.conv5 = nn.Conv2d(N_CHANNELS*64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.relu5 = nn.ReLU()
            self.bn5 = nn.BatchNorm2d(128)
            init.kaiming_normal_(self.conv5.weight, a=0.1)
            self.conv5.bias.data.zero_()
            conv_layers += [self.conv5, self.relu5, self.bn5]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        if layers == 4:
            self.lin = nn.Linear(in_features=64, out_features=len(ID_list))
        if layers == 5:
            self.lin = nn.Linear(in_features=128, out_features=len(ID_list))
        self.dropout = nn.Dropout(dropout)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Augmentation if activated
        if augment_train:
            mix_ratio = random.uniform(0, 1)                                # generate mix ratio for IRs
            x_noise = self.noise(x)                                         # apply noise to signals
            x_impulse = self.impulse(x_noise)                               # apply IR to signals
            x = (mix_ratio * x_noise) + ((1 - mix_ratio) * x_impulse)       # mix noise and IR signals

        # Mel frontend if activated
        if frontend == "MEL":
            x = self.spectrogram(x)
            x = self.amp_db(x)

        # Leaf frontend if activated
        if frontend == "LEAF":
            x = self.leaf(x)
            x = x[:, None, :, :]                                            # add dummy dimension to fit next stage

        # leafPCEN frontend if activated
        if frontend == "leafPCEN":
            self.leaf._complex_conv._kernel.requires_grad = False           # disable training of filters and pooling
            self.leaf._pooling.weights.requires_grad = False
            self.leaf._pooling._bias.requires_grad = False
            x = self.leaf(x)
            x = x[:, None, :, :]                                            # add dummy dimension to fit next stage

        # leafFB frontend if activated
        if frontend == "leafFB":
            self.leaf._compression.alpha.requires_grad = False              # disable training of PCEN
            self.leaf._compression.delta.requires_grad = False
            self.leaf._compression.root.requires_grad = False
            self.leaf._compression.ema._weights.requires_grad = False
            x = self.leaf(x)
            x = x[:, None, :, :]                                            # add dummy dimension to fit next stage

        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        x = self.dropout(x)

        return x


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
myModel = myModel.to(device)


# training and validation
def training(model, train_dl, num_epochs):
    global checkpoint
    global augment_train
    checkpoint = 0
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs, anneal_strategy='linear')
    # performance tracking parameters
    train_losses = []
    train_scores = []
    val_scores = []
    val_losses = []

    # early stop parameters
    last_loss = 0
    patience = 0

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        model.train()

        if augmentation:
            augment_train = True
        else:
            augment_train = False

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            torch.cuda.empty_cache()
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        avg_train_loss = running_loss / len(train_dl)
        train_losses.append(avg_train_loss)
        train_acc = correct_prediction / total_prediction
        train_scores.append(train_acc)
        print(f'E{epoch+1} Train Loss: {avg_train_loss:.2f}, Accuracy: {train_acc:.2f}')

        # Disable gradient updates for validation
        with torch.no_grad():
            model.eval()
            correct_val_prediction = 0
            total_val_prediction = 0
            running_val_loss = 0.0
            augment_train = False       # disable augmentation of validation dataset

            for data in validation_dl:
                torch.cuda.empty_cache()
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                # Keep stats for Loss and Accuracy
                running_val_loss += val_loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)

                # Count of predictions that matched the target label
                correct_val_prediction += (prediction == labels).sum().item()
                total_val_prediction += prediction.shape[0]

        # Print stats at the end of the validation step
        val_acc = correct_val_prediction / total_val_prediction
        val_scores.append(val_acc)
        avg_val_loss = running_val_loss / len(validation_dl)
        val_losses.append(avg_val_loss)

        if not early_stop:
            print(f"E{epoch + 1} Valid Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}")
            checkpoint = epoch + 1

        # Early stopping
        if early_stop:
            if epoch == 0:                      # first epoch
                last_loss = avg_val_loss
                checkpoint = epoch + 1          # save model
                torch.save(model, os.getcwd() + f"/Results/{frontend}_{seed}_{checkpoint}.pth")

                print(f"E{epoch + 1} Valid Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                      f"Patience: {patience}/{max_patience}")

            if epoch > 0:                                               # after first epoch
                if avg_val_loss <= last_loss:                           # if loss improved
                    patience = 0                                        # reset patience
                    last_loss = avg_val_loss                            # update ideal loss
                    os.remove(os.getcwd() + f"/Results/{frontend}_{seed}_{checkpoint}.pth")
                    checkpoint = epoch + 1                              # delete old model state, save new model
                    torch.save(model, os.getcwd() + f"/Results/{frontend}_{seed}_{checkpoint}.pth")

                    print(f"E{epoch + 1} Valid Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                          f"Patience: {patience}/{max_patience}")

                if avg_val_loss > last_loss:                            # if loss does not improve
                    patience += 1                                       # update patience
                    print(f"E{epoch + 1} Valid Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                          f"Patience: {patience}/{max_patience}")

                    if patience == max_patience:                        # if maximum patience is reached
                        print("Stop Training")

                        # plot model training performance
                        fig, axs = plt.subplots(2, 1, sharex="all")
                        axs[0].plot(range(1, len(train_scores) + 1), train_scores, label="Train")
                        axs[0].plot(range(1, len(val_scores) + 1), val_scores, label="Validation")
                        axs[0].legend(loc="lower right")
                        axs[0].set_ylabel("Accuracy")
                        axs[1].plot(range(1, len(train_losses) + 1), train_losses)
                        axs[1].plot(range(1, len(val_losses) + 1), val_losses)
                        axs[1].set_xlabel("Epoch")
                        axs[1].set_ylabel("Loss")
                        fig.suptitle(f"{frontend}_{seed}_{checkpoint}")
                        fig.subplots_adjust(hspace=.001)
                        axs[1].xaxis.set_major_locator(MultipleLocator(5))
                        axs[1].xaxis.set_major_formatter('{x:.0f}')
                        axs[1].xaxis.set_minor_locator(MultipleLocator(1))
                        plt.axvline(x=checkpoint, linestyle="dashed", color="r")
                        axs[0].title.set_visible(False)
                        plt.savefig(os.getcwd() + f"/Results/Perf_{frontend}_{seed}_{checkpoint}.pdf")

                        break

    if not early_stop:
        # plot model training performance
        fig, axs = plt.subplots(2, 1, sharex="all")
        axs[0].plot(range(1, len(train_scores) + 1), train_scores, label="Train")
        axs[0].plot(range(1, len(val_scores) + 1), val_scores, label="Validation")
        axs[0].legend(loc="lower right")
        axs[0].set_ylabel("Accuracy")
        axs[1].plot(range(1, len(train_losses) + 1), train_losses)
        axs[1].plot(range(1, len(val_losses) + 1), val_losses)
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        fig.suptitle(f"{frontend}_{seed}_{checkpoint}")
        fig.subplots_adjust(hspace=.001)
        axs[1].xaxis.set_major_locator(MultipleLocator(5))
        axs[1].xaxis.set_major_formatter('{x:.0f}')
        axs[1].xaxis.set_minor_locator(MultipleLocator(1))
        axs[0].title.set_visible(False)
        plt.savefig(os.getcwd() + f"/Results/Perf_{frontend}_{seed}_{checkpoint}.pdf")

    return checkpoint


# test evaluation
def inference(model, test_dl):
    global augment_train
    correct_test_prediction = 0
    total_test_prediction = 0
    model.eval()
    augment_train = False           # disable augmentation of test dataset

    # for confusion matrix
    y_pred = []
    y_true = []

    # Disable gradient updates
    with torch.no_grad():
        counter = 0
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # record all true labels
            if counter == 0:
                label_list = labels
            else:
                label_list = torch.cat((label_list, labels), dim=0)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # record all predicted labels
            if counter == 0:
                prediction_list = prediction
                counter += 1
            else:
                prediction_list = torch.cat((prediction_list, prediction), dim=0)

            # Count of predictions that matched the target label
            correct_test_prediction += (prediction == labels).sum().item()
            total_test_prediction += prediction.shape[0]

            # for confusion matrix
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            cm_label = labels.data.cpu().numpy()
            y_true.extend(cm_label)  # Save Truth

    # calculate accuracy, f1 score, precision and recall
    test_acc = correct_test_prediction / total_test_prediction
    f1 = f1_score(label_list.cpu(), prediction_list.cpu(), average="macro")
    precision = precision_score(label_list.cpu(), prediction_list.cpu(), average="macro", zero_division=0)
    recall = recall_score(label_list.cpu(), prediction_list.cpu(), average="macro")

    print(f'Test Accuracy: {test_acc:.2f}, F1 score: {f1:.2f}, '
          f'Recall: {recall:.2f}, Precision: {precision:.2f}')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true", labels=classes_list)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes_dict],
                         columns=[i for i in classes_dict])

    plt.figure(figsize=(30, 30))
    plt.subplots_adjust(bottom=0.19, left=0.19)
    plt.title(f"{frontend}_{seed}_{checkpoint}")
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    sn.heatmap(df_cm, annot=True, fmt=".2f", cbar=True, square=True, linewidths=0.2, linecolor="dimgrey")
    plt.savefig(os.getcwd() + f"/Results/CM_{frontend}_{seed}_{checkpoint}.pdf")


# train model if specified
if train_model:
    training(myModel, train_dl, num_epochs)
    # save fully trained model if early stopping is deactivated
    if not early_stop:
        torch.save(myModel, os.getcwd() + f"/Results/{frontend}_{seed}_{checkpoint}.pth")

if not train_model:
    if test_model:
        checkpoint = test_point         # specify for testing saved model

ModelTest = f"/Results/{frontend}_{seed}_{checkpoint}.pth"

# load and test saved model if specified
if test_model:
    myModel = torch.load(os.getcwd() + ModelTest, map_location=torch.device('cpu'))
    inference(myModel, test_dl)
