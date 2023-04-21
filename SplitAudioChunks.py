import librosa
import soundfile as sf
import os
import glob
import ntpath
import shutil
import numpy as np
import warnings

# disable FutureWarning output
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
dataset = "66 Train"              # dataset to be split
SR = 44100                          # sample rate
chunk_length = 5                    # length of chunks in seconds
overlap = 3.75                      # length of overlap between consecutive chunks
min_length = 1.25                   # minimum length for end chunks to be created

# calculate global variables
buffer = chunk_length * SR          # number of samples per chunk
overlap_samples = overlap * SR      # overlap of chunks in samples
min_samples = min_length * SR       # minimum end samples

# input path
base_path = "/Users/mariusfaiss/PycharmProjects/TDSproject/Training/Datasets"
path = f"{base_path}/{dataset}/"

# create new folder and/or delete old files
folder = f"{base_path}/{dataset} Chunks/"
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

# process all files in input folder
for filename in glob.glob(os.path.join(path, '*.wav')):                     # wav files only
    audio, sr = librosa.load(filename, sr=SR)                               # load file with correct SR
    file_length = librosa.get_duration(audio, sr=SR)                        # get file length in seconds
    name = ntpath.basename(filename[:-4])                                   # filename without path or extension
    print("Processing: " + name)

    # Calculate and initialize local variables
    samples_total = file_length * SR                                        # overall n samples per file
    samples_wrote = 0                                                       # initialize start time

    # loop files that are shorter than chunk length
    if samples_total < buffer:
        count = int(buffer / samples_total) + (buffer % samples_total > 0)  # rounded how often the file fits in buffer
        i = 1                                                               # init counter
        loop = audio                                                        # init loop

        while i < count:
            loop = np.concatenate([loop, audio])                            # add file to itself until buffer is filled
            i += 1                                                          # update counter

        loop = loop[0: buffer]                                              # truncate loop to specified chunk length
        out_filename = folder + name + "_loop" + ".wav"                     # create output file name
        sf.write(out_filename, loop, SR)                                    # save file

    # split longer files into chunks
    if file_length >= chunk_length:
        counter = 1                                                                 # initialize counter for file name
        while samples_wrote < samples_total:
            if (samples_total - samples_wrote) >= buffer:                           # if buffer fits in remaining time
                chunk = audio[samples_wrote: (samples_wrote + buffer)]              # create the audio chunk
                out_filename = folder + name + "_chunk" + str(counter) + ".wav"     # create file name with counter
                sf.write(out_filename, chunk, SR)                                   # export file
                counter += 1                                                        # update counter
                samples_wrote = int(samples_wrote + buffer - overlap_samples)       # update start position

            # wrap audio for end chunks
            if (samples_total - samples_wrote) < buffer:                            # if remaining time is too short...
                if (samples_total - samples_wrote) > min_samples:                   # ... and longer than minimum time
                    wrap_length = int(buffer - (samples_total - samples_wrote))     # wrap length
                    wrap = audio[0: wrap_length]                                    # create wrap chunk
                    chunk = audio[samples_wrote: (samples_wrote + buffer)]          # create the audio chunk
                    wrapped_file = np.concatenate([chunk, wrap])                    # combine chunk with wrap chunk
                    out_filename = folder + name + "_wrap" + str(counter) + ".wav"  # file name
                    sf.write(out_filename, wrapped_file, SR)                        # export file
                    counter += 1                                                    # update counter
                samples_wrote = int(samples_wrote + buffer - overlap_samples)       # update start position
