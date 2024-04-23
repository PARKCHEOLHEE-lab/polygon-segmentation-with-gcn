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

    IMG_QA_PATH = os.path.abspath(os.path.join(SAVE_PATH, "qa"))
    IMG_QA_NAME_REGULAR = "lands_gdf_regular.png"
    IMG_QA_NAME_IRREGULAR = "lands_gdf_irregular.png"

    LAND_AREA_THRESHOLD = 500

    THRESHOLD_MRR_RATIO_REGULAR = 0.83
    THREHSOLD_MRR_RATIO_IRREGULAR_MAX = 0.53
    THREHSOLD_MRR_RATIO_IRREGULAR_MIN = 0.4
    THREHSOLD_INNDER_DEGREE_SUM_IRREGULAR = 850

    SIMPLIFICATION_DEGREE = 15.0


class ModelConfiguration:
    pass


class Configuration(DataConfiguration, ModelConfiguration):
    pass
