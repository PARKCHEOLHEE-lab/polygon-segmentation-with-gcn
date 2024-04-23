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

    USAGE_ROAD_1 = "도로"
    USAGE_ROAD_2 = "도로등"

    USAGE_WATERWAY_1 = "하천"
    USAGE_WATERWAY_2 = "하천등"

    USAGE_PARK_1 = "공원"
    USAGE_PARK_2 = "공원등"
