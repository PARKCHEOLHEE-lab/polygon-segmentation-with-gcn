import os


class DataConfiguration:
    TOLERANCE = 1e-6
    TOLERANCE_MACRO = 1e-2
    TOLERANCE_LARGE = 1e-4
    TOLEARNCE_DEGREE = 0.1
    TOLERANCE_CENTERLINE = 0.2

    SEGMENT_DIVIDE_BASELINE = 0.2
    EVEN_AREA_WEIGHT = 0.34
    OMBR_RATIO_WEIGHT = 0.67
    SLOPE_SIMILARITY_WEIGHT = 0.045

    TOTAL_LANDS_FOLDER_COUNT = 19

    SHP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/shp"))
    SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))
    SAVE_RAW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))

    LANDS_ZIP_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands_zip"))
    LANDS_PATH = os.path.abspath(os.path.join(SHP_PATH, "lands"))

    LANDS_GDF_REGULAR_NAME = "lands_gdf_regular"
    LANDS_GDF_IRREGULAR_NAME = "lands_gdf_irregular"

    LANDS_GDF_REGULAR_PKL = f"{LANDS_GDF_REGULAR_NAME}.pkl"
    LANDS_GDF_IRREGULAR_PKL = f"{LANDS_GDF_IRREGULAR_NAME}.pkl"

    LANDS_GDF_REGULAR_PNG = f"{LANDS_GDF_REGULAR_NAME}.png"
    LANDS_GDF_IRREGULAR_PNG = f"{LANDS_GDF_IRREGULAR_NAME}.png"

    LAND_AREA_THRESHOLD = 100

    THRESHOLD_MRR_RATIO_REGULAR = 0.83
    THRESHOLD_MRR_RATIO_IRREGULAR_MAX = 0.60
    THRESHOLD_MRR_RATIO_IRREGULAR_MIN = 0.10
    THRESHOLD_INNDER_DEGREE_SUM_IRREGULAR = 850

    REGULAR_NUMBER_TO_GENERATE = 2000
    IRREGULAR_NUMBER_TO_GENERATE = 2000

    SIMPLIFICATION_DEGREE = 10.0


class ModelConfiguration:
    pass


class Configuration(DataConfiguration, ModelConfiguration):
    pass
