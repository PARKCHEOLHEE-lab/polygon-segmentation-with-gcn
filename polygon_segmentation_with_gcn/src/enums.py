class LandShape:
    """All shapes:
    가로장방
    세로장방
    사다리형
    정방형
    자루형
    지정되지않음
    부정형
    """

    SHAPE_HORIZONTAL_RECTANGLE = "가로장방"
    SHAPE_VERTICAL_RECTANGLE = "세로장방"
    SHAPE_LADDER = "사다리형"
    SHAPE_SQUARE = "정방형"
    SHAPE_FLAG = "자루형"
    SHAPE_UNDEFINED = "지정되지않음"
    SHAPE_IRREGULARITY = "부정형"


class LandUsage:
    """
    All usages:
        도로등,
        주거기타,
        상업용,
        연립,
        하천등,
        상업나지,
        상업기타,
        업무용,
        주상나지,
        주상용,
        공원등,
        주상기타,
        다세대,
        주차장등,
        아파트,
        주거나지,
        자연림,
        물류터미널,
        운동장등,
        위험시설,
        임야기타,
        단독,
        공업용
    """

    USAGE_ROAD_1 = "도로등"
    USAGE_ROAD_2 = "도로"
    USAGE_WILDWOOD = "자연림"
    USAGE_PARK = "공원등"
    USAGE_PARKING = "주차장등"
    USAGE_FOREST = "임야기타"
    USAGE_HOUSING_EMPTY = "주상나지"
    USAGE_COMMERCIAL_EMPTY = "상업나지"
    USAGE_TERMINAL = "물류터미널"
    USAGE_STADIUM = "운동장등"
    USAGE_DANGEROUS_FACILITY = "위험시설"
    USAGE_WATERWAY = "하천등"
    USAGE_INDUSTRY = "공업용"

    USAGE_TO_EXCLUDE = [
        USAGE_ROAD_1,
        USAGE_WILDWOOD,
        USAGE_PARK,
        USAGE_PARKING,
        USAGE_FOREST,
        USAGE_HOUSING_EMPTY,
        USAGE_COMMERCIAL_EMPTY,
        USAGE_TERMINAL,
        USAGE_STADIUM,
        USAGE_DANGEROUS_FACILITY,
        USAGE_WATERWAY,
        USAGE_INDUSTRY,
    ]
