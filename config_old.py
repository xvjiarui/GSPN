import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # NUM_CATEGORY = 100 # NYUv2: 1+35 +1? # ScanNet: 1+18 #(background+classes)
    NUM_CATEGORY = 2 # NYUv2: 1+35 +1? # ScanNet: 1+18 #(background+classes)
    NUM_GROUP = 100
    NUM_POINT = 10000 # 8192 # ScanNet: 18000
    # Num of points for each generated instance
    NUM_POINT_INS = 512
    BATCH_SIZE = 2
    # How many seed points to sample for generation
    NUM_SAMPLE = 1024 # 512 # 2048 # 2048 # 1024 #

    # ROIs kept after sorting and before non-maximum suppression
    SPN_PRE_NMS_LIMIT = 1536 # 192 # 1536 # 384 # 768 # 
    # ROIs kept after non-maximum suppression (training and inference)
    SPN_NMS_MAX_SIZE_TRAINING = 128 # 512 # 256 # 512 # 
    SPN_NMS_MAX_SIZE_INFERENCE = 1536 # 96 # 384 # 192 # 384 # 
    SPN_IOU_THRESHOLD = 0.8 # default 0.5
    SPN_SCORE_THRESHOLD = float('-inf')

    NUM_POINT_INS_MASK = 512 # 256 # 1024 # 1024 # 512 # 
    TRAIN_ROIS_PER_IMAGE = 64 # 512 # 256 # 128 # 
    ROI_POSITIVE_RATIO = 0.33
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    NORMALIZE_CROP_REGION = True

    SHRINK_BOX = False # Default: False
    USE_COLOR = False
    TRAIN_MODULE = ['SPN'] # Option: ['SPN', 'MRCNN']

    DETECTION_MIN_CONFIDENCE = 0.1 # Default: 0.3
    DETECTION_NMS_THRESHOLD = 0.1 # Default: 0.1
    DETECTION_MAX_INSTANCES = 200

    ## image size for NYUv2
    IMG_H = 316
    IMG_W = 415


    def __init__(self):
        """Set values of computed attributes."""

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
