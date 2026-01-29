## Keyword Spotting Model

Training data from <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz> and should be put in a directory speech-data\
Training data in LJSpeech is from https://keithito.com/LJ-Speech-Dataset/
`audio_preprocessor.py` has the MFCC computation\
`augment_data.py` generates training data by augmenting Google's dataset with noise\
`sort_data.py` creates txt files with file paths for training, testing, and validation datasets\
`train.py` uses all of these to train a model on multiple generated datasets\
trained model params from running `train.py` are in `./out`\
`nn.py` has the model and data loader. Run standalone it will train the model using the training data set listed in `training_set.txt`\
trained model params from running `nn.py` are in `model.pth`\
`audio_stream.py` opens a streaming mic to detect the "stop" keyword