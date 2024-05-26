import os
import torch
import random
import numpy as np


class DataConfiguration:
    TOLERANCE = 1e-6
    TOLERANCE_MACRO = 1e-2
    TOLERANCE_LARGE = 1e-4
    TOLEARNCE_DEGREE = 0.1
    TOLERANCE_CENTERLINE = 0.2

    SEGMENT_DIVIDE_BASELINE_TO_TRIANGULATE = 0.2
    SEGMENT_DIVIDE_BASELINE_TO_POLYGON = 0.4
    EVEN_AREA_WEIGHT = 0.34
    OMBR_RATIO_WEIGHT = 0.67
    SLOPE_SIMILARITY_WEIGHT = 0.045

    TOTAL_LANDS_FOLDER_COUNT = 19

    SHP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/shp"))
    SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))
    SAVE_RAW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))

    MERGED_SAVE_PATH = os.path.abspath(os.path.join(SAVE_PATH, "merged"))

    LANDS_ZIP_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands_zip"))
    LANDS_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands"))

    LANDS_GDF_REGULAR_NAME = "lands_gdf_regular"
    LANDS_GDF_IRREGULAR_NAME = "lands_gdf_irregular"

    LANDS_GDF_REGULAR_PKL = f"{LANDS_GDF_REGULAR_NAME}.pkl"
    LANDS_GDF_IRREGULAR_PKL = f"{LANDS_GDF_IRREGULAR_NAME}.pkl"

    LANDS_GDF_REGULAR_PNG = f"{LANDS_GDF_REGULAR_NAME}.png"
    LANDS_GDF_IRREGULAR_PNG = f"{LANDS_GDF_IRREGULAR_NAME}.png"

    LANDS_DATA_IRREGULAR_NAME = "lands_data_irregular"
    LANDS_DATA_IRREGULAR_PT = f"{LANDS_DATA_IRREGULAR_NAME}.pt"

    LANDS_DATA_REGULAR_NAME = "lands_data_regular"
    LANDS_DATA_REGULAR_PT = f"{LANDS_DATA_REGULAR_NAME}.pt"

    LAND_AREA_THRESHOLD = 100

    THRESHOLD_MRR_RATIO_REGULAR = 0.83
    THRESHOLD_MRR_RATIO_IRREGULAR_MAX = 0.60
    THRESHOLD_MRR_RATIO_IRREGULAR_MIN = 0.10
    THRESHOLD_INNDER_DEGREE_SUM_IRREGULAR = 850

    REGULAR_NUMBER_TO_GENERATE = 2000
    IRREGULAR_NUMBER_TO_GENERATE = 2000

    SIMPLIFICATION_DEGREE = 10.0

    ROTATION_DEGREE_MAX = 360
    ROTATION_INTERVAL = 18.0

    TRAIN_SPLIT_RATIO = 0.75
    VALIDATION_SPLIT_RATIO = 0.15
    TEST_SPLIT_RATIO = 0.1
    SPLIT_RATIOS = [TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO]


class ModelConfiguration:
    EPOCH = 300

    DROPOUT_RATE = 0.5
    BATCH_SIZE = 64
    HIDDEN_CHANNELS = 256
    OUT_CHANNELS = 64
    LEARNING_RATE = 0.00002

    CONNECTIVITY_THRESHOLD = 0.5
    NEGATIVE_SAMPLE_MULTIPLIER = 1


class Configuration(DataConfiguration, ModelConfiguration):
    def __iter__(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("_"):
                yield attr, getattr(self, attr)

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    DEFAULT_SEED = 777
    SEED_SET = None

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))
    STATES_PTH = "states.pth"

    @staticmethod
    def set_seed(seed: int = DEFAULT_SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("CUDA status")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  DEVICE: {Configuration.DEVICE} \n")

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED_SET = seed
