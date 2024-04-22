import os


class DataConfiguration:
    TOLERANCE = 1e-6
    TOLERANCE_LARGE = 1e-4
    TOLEARNCE_DEGREE = 0.1

    EVEN_AREA_WEIGHT = 0.34
    OMBR_RATIO_WEIGHT = 0.67
    SLOPE_SIMILARITY_WEIGHT = 0.045

    SHP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/shp"))
    SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))

    LANDS_ZIP_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands_zip"))
    LANDS_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands"))

    APT_STRING = "아파트"
    APT_LAND_AREA_THRESHOLD = 1000
    MRR_RATIO_THRESHOLD = 0.5


class ModelConfiguration:
    pass


class Configuration(DataConfiguration, ModelConfiguration):
    pass
