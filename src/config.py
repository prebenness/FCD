'''
Global config variables, global read/write access
Sets default values for optional variables
'''

# General settings
MODE = None
DEVICE = 'cuda'
MULTI_GPU = False

# Dataset
DATASET = None
NUM_CHANNELS = None
HEIGHT = None
WIDTH = None
NUM_CLASSES = None

# Training/eval/test settings
BATCH_SIZE = 128
NUM_EPOCHS = 10
