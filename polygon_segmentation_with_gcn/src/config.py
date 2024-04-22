import os


class DataConfiguration:
    TOLERANCE = 1e-6
    TOLERANCE_LARGE = 1e-4
    TOLEARNCE_DEGREE = 0.1

    EVEN_AREA_WEIGHT = 0.34
    OMBR_RATIO_WEIGHT = 0.67
    SLOPE_SIMILARITY_WEIGHT = 0.045

    TOTAL_LANDS_FOLDER_COUNT = 19

    SHP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/shp"))
    SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))

    LANDS_ZIP_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands_zip"))
    LANDS_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands"))

    LAND_AREA_THRESHOLD = 500

    MRR_RATIO_THRESHOLD_REGULAR = 0.83
    MRR_RATIO_THRESHOLD_IRREGULAR = 0.7


class ModelConfiguration:
    pass


class Configuration(DataConfiguration, ModelConfiguration):
    pass
