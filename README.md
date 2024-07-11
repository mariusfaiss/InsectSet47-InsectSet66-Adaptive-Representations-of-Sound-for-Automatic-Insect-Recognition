# InsectSet47-InsectSet66-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition

This repository contains the code used for a paper published in [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1011541) comparing the waveform-based frontend [LEAF](https://github.com/google-research/leaf-audio) to the classic mel-spectrogram approach for classifying insect sounds. The paper is submitted for publication and a preprint is publicly available. The datasets that were compiled for this work are uploaded on zenodo.org ([InsectSet32](https://zenodo.org/record/7072196), [InsectSet47&InsectSet66](https://zenodo.org/record/7828439)). The code for the experiments on [InsectSet32](https://zenodo.org/record/7072196) is published separately on [Github](https://github.com/mariusfaiss/InsectSet32-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition).

This repository includes the [classifier](https://github.com/mariusfaiss/InsectSet47-InsectSet66-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/LEAF_Mel_Model.py) that was built and tested for [InsectSet47&InsectSet66](https://zenodo.org/records/8252141), which can be used with a mel-spectrogram frontend as the standard approach, or with the adaptive, waveform based frontend [LEAF](https://github.com/google-research/leaf-audio) (using the [pytorch implementation](https://github.com/SarthakYadav/leaf-pytorch)) which generally achieved higher performance. For preparation of the input data, a [script that splits](https://github.com/mariusfaiss/InsectSet47-InsectSet66-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/SplitAudioChunks.py) the audio files into overlapping five second long chunks of audio is included. This should be applied to all audio files to match the input length of the classifier.

Below is the abstract of the preprint describing the project:

Insect population numbers and biodiversity have been rapidly declining with time, and monitoring these trends has become increasingly important for conservation measures to be effectively implemented. But monitoring methods are often invasive, time and resource intense, and prone to various biases. Many insect species produce characteristic sounds that can easily be detected and recorded without large cost or effort. Using deep learning methods, insect sounds from field recordings could be automatically detected and classified to monitor biodiversity and species distribution ranges. We implement this using recently published datasets of insect sounds (Orthoptera and Cicadidae) and machine learning methods and evaluate their potential for acoustic insect monitoring. We compare the performance of the conventional spectrogram-based audio representation against LEAF, a new adaptive and waveform-based frontend. LEAF achieved better classification performance than the mel-spectrogram frontend by adapting its feature extraction parameters during training. This result is encouraging for future implementations of deep learning technology for automatic insect sound recognition, especially as larger datasets become available.

The IR files used for data augmentation are sourced from the [OpenAIR](https://www.openairlib.net) library.

[Gill Heads Mine](https://www.openair.hosted.york.ac.uk/?page_id=494)

44100_dales_site1_4way_mono.wav

44100_dales_site2_4way_mono.wav

44100_dales_site3_4way_mono.wav

[Koli National Park - Winter](https://www.openair.hosted.york.ac.uk/?page_id=584)

44100_koli_snow_site1_4way_mono.wav

44100_koli_snow_site2_4way_mono.wav

44100_koli_snow_site3_4way_mono.wav

44100_koli_snow_site4_4way_mono.wav

[Koli National Park - Summer](https://www.openair.hosted.york.ac.uk/?page_id=577)

44100_koli_summer_site1_4way_mono.wav

44100_koli_summer_site2_4way_mono.wav

44100_koli_summer_site3_4way_mono.wav

44100_koli_summer_site4_4way_mono.wav
