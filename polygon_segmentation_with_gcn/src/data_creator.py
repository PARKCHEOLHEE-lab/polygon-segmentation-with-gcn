import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import io
import shapely
import traceback
import pygeoops
import itertools
import numpy as np
import multiprocessing
import geopandas as gpd
import zipfile36 as zipfile
import matplotlib.pyplot as plt


from PIL import Image
from typing import List, Tuple
from shapely import ops, affinity
from src import commonutils, enums
from src.config import DataConfiguration
from shapely.geometry import (
    Point,
    MultiPoint,
    Polygon,
    MultiPolygon,
    MultiLineString,
    LineString,
    CAP_STYLE,
    JOIN_STYLE,
)


class DataCreatorHelper:
    @staticmethod
    def divide_linestring(linestring: LineString, count_to_divide: int) -> List[Point]:
        """_summary_

        Args:
            linestring (LineString): _description_
            count_to_divide (int): _description_

        Returns:
            List[Point]: _description_
        """

        linestring_coordinates = np.array(linestring.coords)

        assert len(linestring_coordinates) == 2, "Only can straight linestring be divided."

        linestring_vector = linestring_coordinates[1] - linestring_coordinates[0]
        linestring_vector = linestring_vector / np.linalg.norm(linestring_vector)
        linestring_divider = linestring.length / count_to_divide

        divided_points = []
        for count in range(count_to_divide + 1):
            divided_points.append(Point(linestring_coordinates[0] + linestring_vector * linestring_divider * count))

        return divided_points

    @staticmethod
    def extend_linestring(linestring: LineString, start: float, end: float) -> LineString:
        """Extend a given linestring by the given `start` and `end` values

        Args:
            linestring (LineString): linestring
            start (float): start value
            end (float): end value

        Returns:
            LineString: extended linestring
        """

        linestring_coordinates = np.array(linestring.coords)

        assert len(linestring_coordinates) == 2, "Only can straight linestring be extended."

        a, b = linestring_coordinates

        ab = b - a
        ba = a - b

        ab_normalized = ab / np.linalg.norm(ab)
        ba_normalized = ba / np.linalg.norm(ba)

        a_extended = a + ba_normalized * start
        b_extended = b + ab_normalized * end

        extended_linestring = LineString([a_extended, b_extended])

        assert np.isclose(extended_linestring.length, linestring.length + start + end), "Extension failed."

        return extended_linestring

    @staticmethod
    def compute_slope(p1: np.ndarray, p2: np.ndarray) -> float:
        if np.isclose(p2[0] - p1[0], 0):
            return -np.inf

        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    @staticmethod
    def compute_polyon_degrees(polygon: Polygon, return_sum: bool = False) -> List[float]:
        """Compute polygon degrees

        Args:
            polygon (Polygon): polygon

        Returns:
            List[float]: polygon degrees
        """

        exterior_coordinates = polygon.exterior.coords[:-1]

        polygon_degrees = []

        for ci in range(len(exterior_coordinates)):
            a = np.array(exterior_coordinates[ci])
            b = np.array(exterior_coordinates[ci - 1])
            c = np.array(exterior_coordinates[(ci + 1) % len(exterior_coordinates)])

            ab = b - a
            ac = c - a

            cos_theta = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
            angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle_degrees = np.degrees(angle_radians)

            polygon_degrees.append(angle_degrees)

        if return_sum:
            return sum(polygon_degrees)

        return polygon_degrees

    @staticmethod
    def simplify_polygon(polygon: Polygon, tolerance_degree: float = DataConfiguration.TOLEARNCE_DEGREE) -> Polygon:
        """Simplify a given polygon by removing vertices

        Args:
            polygon (Polygon): polygon
            tolerance_degree (float, optional): threshold to remove. Defaults to 1.0.

        Returns:
            Polygon: _description_
        """

        exterior_coordinates = polygon.exterior.coords
        if np.allclose(exterior_coordinates[0], exterior_coordinates[-1]):
            exterior_coordinates = exterior_coordinates[:-1]

        polygon_degrees = DataCreatorHelper.compute_polyon_degrees(polygon)

        assert len(exterior_coordinates) == len(polygon_degrees), "Lengths condition is not satisfied."

        simplified_coordinates = []
        for degree, coord in zip(polygon_degrees, exterior_coordinates):
            if not (180 - tolerance_degree < degree < 180 + tolerance_degree):
                simplified_coordinates.append(coord)

        if len(simplified_coordinates) < 3:
            return polygon

        return Polygon(simplified_coordinates)

    @staticmethod
    def explode_polygon(polygon: Polygon) -> List[LineString]:
        """Explode a given polygon into a list of LineString objects.

        Args:
            polygon (Polygon): polygon

        Returns:
            List[LineString]: polygon segments
        """

        return [
            LineString([polygon.exterior.coords[ci], polygon.exterior.coords[ci + 1]])
            for ci in range(len(polygon.exterior.coords) - 1)
        ]

    @staticmethod
    def insert_vertices_into_polygon(polygon: Polygon, _vertices_to_insert: List[Point]) -> Polygon:
        """Insert given vertices into the given polygon

        Args:
            polygon (Polygon): polygon
            _vertices_to_insert (List[Point]): vertices to insert

        Returns:
            Polygon: polygon with inserted vertices
        """

        polygon_vertices = list(polygon.exterior.coords)

        vertices_to_insert = _vertices_to_insert[:]

        curr_vi = 0
        while curr_vi < len(polygon_vertices):
            is_inserted = False

            next_vi = (curr_vi + 1) % len(polygon_vertices)

            curr_vertex = polygon_vertices[curr_vi]
            next_vertex = polygon_vertices[next_vi]

            for i, ipt in enumerate(vertices_to_insert):
                p1 = np.array(curr_vertex)
                p2 = np.array(next_vertex)
                p3 = np.array(ipt.coords[0])

                a = p2 - p1
                b = p3 - p1

                if not is_inserted and np.isclose(np.cross(a, b), 0) and 0 < np.dot(a, b) < np.dot(a, a):
                    polygon_vertices.insert(next_vi, vertices_to_insert.pop(i).coords[0])
                    is_inserted = True
                    break

            if is_inserted:
                continue

            curr_vi += 1

        inserted_polygon = Polygon(polygon_vertices)

        return inserted_polygon

    @staticmethod
    def insert_intersected_vertices(polygon: Polygon) -> Polygon:
        """_summary_

        Args:
            polygon (Polygon): _description_

        Returns:
            Polygon: _description_
        """

        intersects_pts_to_include = []
        polygon_segments = DataCreatorHelper.explode_polygon(polygon)
        polygon_vertices = MultiPoint(polygon.boundary.coords)

        for segment in polygon_segments:
            extended_segment = affinity.scale(segment, 100, 100)

            ipts = extended_segment.intersection(polygon.boundary)
            if ipts.is_empty:
                continue

            if isinstance(ipts, MultiPoint):
                ipts = list(ipts.geoms)
            else:
                ipts = [ipts]

            for ipt in ipts:
                if not polygon_vertices.buffer(DataConfiguration.TOLERANCE).contains(ipt):
                    if isinstance(ipt, Point):
                        intersects_pts_to_include.append(ipt)

        inserted_polygon = DataCreatorHelper.insert_vertices_into_polygon(polygon, intersects_pts_to_include)

        return inserted_polygon

    @staticmethod
    def normalize_polygon(polygon: Polygon) -> Polygon:
        """Centralize a given polygon to (0, 0) and normalize it to (-1, 1).

        Args:
            polygon (Polygon): polygon to normalize

        Returns:
            Polygon: The normalized polygon with its maximum norm being 1.
        """

        exterior_coordinates = np.array(polygon.exterior.coords)
        centroid = exterior_coordinates.mean(axis=0)

        centralized_coordinates = exterior_coordinates - centroid

        max_norms = np.linalg.norm(centralized_coordinates, axis=1).max()

        normalized_coordinates = centralized_coordinates / max_norms

        return Polygon(normalized_coordinates)

    @staticmethod
    def compute_mrr_ratio(polygon: Polygon) -> float:
        """_summary_

        Args:
            polygon (Polygon): _description_

        Returns:
            float: _description_
        """

        polygon_area = polygon.area
        mrr_area = polygon.minimum_rotated_rectangle.area

        return polygon_area / mrr_area

    @staticmethod
    def erode_and_dilate_polygon(polygon: Polygon, buffer_distance: float) -> Polygon:
        """_summary_

        Args:
            polygon (Polygon): _description_
            buffer_distance (float): _description_

        Returns:
            Polygon: _description_
        """

        eroded = polygon.buffer(-buffer_distance, join_style=JOIN_STYLE.mitre)
        if isinstance(eroded, MultiPolygon):
            eroded = max(eroded.geoms, key=lambda e: e.area)

        dilated = eroded.buffer(buffer_distance, join_style=JOIN_STYLE.mitre)

        return dilated

    @staticmethod
    def dilate_and_erode_polygon(polygon: Polygon, buffer_distance: float) -> Polygon:
        """_summary_

        Args:
            polygon (Polygon): _description_
            buffer_distance (float): _description_

        Returns:
            Polygon: _description_
        """

        dilated = polygon.buffer(buffer_distance, join_style=JOIN_STYLE.mitre)
        eroded = dilated.buffer(-buffer_distance, join_style=JOIN_STYLE.mitre)

        if isinstance(eroded, MultiPolygon):
            eroded = max(eroded.geoms, key=lambda e: e.area)

        if len(eroded.interiors) > 0:
            eroded = Polygon(eroded.exterior.coords)

        return eroded

    @staticmethod
    def get_polygon_centerline(polygon: Polygon, simplification_tolerance: float = 0.0, **kwargs) -> LineString:
        """_summary_

        Args:
            polygon (Polygon): _description_
            simplification_tolerance (float, optional): _description_. Defaults to 0.0.

        Returns:
            LineString: _description_
        """

        try:
            centerline = pygeoops.centerline(polygon, **kwargs)

            if isinstance(centerline, MultiLineString):
                centerline = max(centerline.geoms, key=lambda g: g.length)

            centerline = centerline.simplify(simplification_tolerance)

            return centerline

        except Exception as e:
            print(f"Error encountered: {e}")
            traceback.print_exc()
            return LineString()

    @staticmethod
    def get_polygon_created_by_exterior(polygon: Polygon) -> Polygon:
        """_summary_

        Args:
            polygon (Polygon): _description_

        Returns:
            Polygon: _description_
        """

        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda g: g.area)

        return Polygon(polygon.exterior.coords)


class DataCreator(DataCreatorHelper, DataConfiguration, enums.LandShape, enums.LandUsage):
    def __init__(
        self,
        shp_dir: str,
        save_dir: str,
        number_to_split: int,
        simplification_degree_factor: float = None,
        segment_threshold_length: float = None,
        even_area_weight: float = DataConfiguration.EVEN_AREA_WEIGHT,
        ombr_ratio_weight: float = DataConfiguration.OMBR_RATIO_WEIGHT,
        slope_similarity_weight: float = DataConfiguration.SLOPE_SIMILARITY_WEIGHT,
        is_debug_mode: bool = False,
    ):
        self.shp_dir = shp_dir
        self.save_dir = save_dir
        self.number_to_split = number_to_split
        self.simplification_degree_factor = simplification_degree_factor
        self.segment_threshold_length = segment_threshold_length
        self.even_area_weight = even_area_weight
        self.ombr_ratio_weight = ombr_ratio_weight
        self.slope_similarity_weight = slope_similarity_weight
        self.is_debug_mode = is_debug_mode

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

    def triangulate_polygon(
        self,
        polygon: Polygon,
        segment_threshold_length: float,
    ) -> Tuple[List[Polygon], List[LineString]]:
        """_summary_

        Args:
            polygon (Polygon): _description_
            simplification_degree_factor (float, optional): _description_. Defaults to None.
            segment_threshold_length (float, optional): _description_. Defaults to None.

        Returns:
            Polygon: _description_
        """

        if isinstance(segment_threshold_length, float):
            polygon_segments = DataCreatorHelper.explode_polygon(polygon)

            vertices_to_insert = []
            for segment in polygon_segments:
                if segment.length > segment_threshold_length:
                    divided_points = DataCreatorHelper.divide_linestring(
                        linestring=segment,
                        count_to_divide=np.ceil(segment.length / segment_threshold_length).astype(int),
                    )[1:-1]

                    vertices_to_insert.extend(divided_points)

            polygon = DataCreatorHelper.insert_vertices_into_polygon(polygon, vertices_to_insert)

        buffered_polygon = polygon.buffer(self.TOLERANCE, join_style=JOIN_STYLE.mitre)

        triangulations = [tri for tri in ops.triangulate(polygon) if tri.within(buffered_polygon)]
        triangulations_filtered_by_area = [tri for tri in triangulations if tri.area >= polygon.area * 0.01]
        triangulations_edges = []

        for tri in triangulations_filtered_by_area:
            for e in DataCreatorHelper.explode_polygon(tri):
                if DataCreatorHelper.extend_linestring(e, -self.TOLERANCE_LARGE, -self.TOLERANCE_LARGE).within(polygon):
                    is_already_existing = False
                    for other_e in triangulations_edges:
                        if e.equals(other_e):
                            is_already_existing = True
                            break

                    if not is_already_existing:
                        triangulations_edges.append(e)

        return triangulations_filtered_by_area, triangulations_edges

    def segment_polygon(
        self,
        polygon: Polygon,
        number_to_split: int,
        segment_threshold_length: float,
        even_area_weight: float,
        ombr_ratio_weight: float,
        slope_similarity_weight: float,
    ):
        """_summary_

        Args:
            polygon (Polygon): _description_
            number_to_split (int): _description_
            simplification_degree_factor (float): _description_
            segment_threshold_length (float): _description_
            even_area_weight (float): _description_
            ombr_ratio_weight (float): _description_
            slope_similarity_weight (float): _description_
        """

        _, triangulations_edges = self.triangulate_polygon(
            polygon=polygon,
            segment_threshold_length=segment_threshold_length,
        )

        splitters_selceted = None
        splits_selected = None
        splits_score = None

        for splitters in list(itertools.combinations(triangulations_edges, number_to_split - 1)):
            exterior_with_splitters = ops.unary_union(list(splitters) + self.explode_polygon(polygon))

            exterior_with_splitters = shapely.set_precision(
                exterior_with_splitters, self.TOLERANCE_LARGE, mode="valid_output"
            )

            exterior_with_splitters = ops.unary_union(exterior_with_splitters)

            splits = list(ops.polygonize(exterior_with_splitters))

            if len(splits) != number_to_split:
                continue

            if any(split.area < polygon.area * 0.25 for split in splits):
                continue

            is_acute_angle_in = False
            is_triangle_shape_in = False
            for split in splits:
                split_segments = self.explode_polygon(split)
                splitter_indices = []

                for ssi, split_segment in enumerate(split_segments):
                    if split_segment.length <= self.TOLERANCE_LARGE * 2:
                        continue

                    reduced_split_segment = DataCreatorHelper.extend_linestring(
                        split_segment, -self.TOLERANCE_LARGE, -self.TOLERANCE_LARGE
                    )
                    buffered_split_segment = reduced_split_segment.buffer(self.TOLERANCE, cap_style=CAP_STYLE.flat)

                    if buffered_split_segment.intersects(MultiLineString(splitters)):
                        splitter_indices.append(ssi)
                        splitter_indices.append(ssi + 1)

                degrees = self.compute_polyon_degrees(split)
                degrees += [degrees[0]]

                if (np.array(degrees)[splitter_indices] < 20).sum():
                    is_acute_angle_in = True
                    break

                if len(self.explode_polygon(self.simplify_polygon(split))) == 3:
                    is_triangle_shape_in = True
                    break

            if is_acute_angle_in or is_triangle_shape_in:
                continue

            sorted_splits_area = sorted([split.area for split in splits], reverse=True)
            even_area_score = (sorted_splits_area[0] - sum(sorted_splits_area[1:])) / polygon.area * even_area_weight

            ombr_ratio_scores = []
            slope_similarity_scores = []

            for split in splits:
                ombr = split.minimum_rotated_rectangle
                each_ombr_ratio = split.area / ombr.area
                inverted_ombr_score = 1 - each_ombr_ratio
                ombr_ratio_scores.append(inverted_ombr_score)

                slopes = []
                for splitter in splitters:
                    if split.buffer(self.TOLERANCE_LARGE).intersects(splitter):
                        slopes.append(self.compute_slope(splitter.coords[0], splitter.coords[1]))

                splitter_main_slope = max(slopes, key=abs)

                split_slopes_similarity = []
                split_segments = self.explode_polygon(split)
                for split_seg in split_segments:
                    split_seg_slope = self.compute_slope(split_seg.coords[0], split_seg.coords[1])
                    split_slopes_similarity.append(abs(splitter_main_slope - split_seg_slope))

                avg_slope_similarity = sum(split_slopes_similarity) / len(split_slopes_similarity)
                slope_similarity_scores.append(avg_slope_similarity)

            ombr_ratio_score = abs(ombr_ratio_scores[0] - sum(ombr_ratio_scores[1:])) * ombr_ratio_weight
            slope_similarity_score = sum(slope_similarity_scores) / len(splits) * slope_similarity_weight

            score_sum = even_area_score + ombr_ratio_score + slope_similarity_score

            if splits_score is None or splits_score > score_sum:
                splits_score = score_sum
                splits_selected = splits
                splitters_selceted = splitters

        return splits_selected, triangulations_edges, splitters_selceted

    @commonutils.runtime_calculator
    def _get_initial_lands_gdf(self, original_lands_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """_summary_

        Args:
            original_lands_gdf (gpd.GeoDataFrame): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """

        lands_gdf = original_lands_gdf[
            (original_lands_gdf.geometry.area >= self.LAND_AREA_THRESHOLD)
            & (~original_lands_gdf.iloc[:, 18].isna())
            & (original_lands_gdf.iloc[:, 11] != self.USAGE_ROAD_1)
            & (original_lands_gdf.iloc[:, 11] != self.USAGE_WATERWAY_1)
            & (original_lands_gdf.iloc[:, 18] != self.USAGE_ROAD_2)
            & (original_lands_gdf.iloc[:, 18] != self.USAGE_WATERWAY_2)
            & (original_lands_gdf.iloc[:, 18] != self.USAGE_DITCH)
        ]

        # 1. Remove holes and multipolygon
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["geometry"] = pool.map(self.get_polygon_created_by_exterior, lands_gdf.geometry.tolist())

        # 2. Normalize geometry
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["normalized_geometry"] = pool.map(self.normalize_polygon, lands_gdf.geometry.tolist())

        # 3. Erode and dilate to tidy
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["buffered_geometry"] = pool.starmap(
                self.erode_and_dilate_polygon,
                [(normalized_geometry, 0.15) for normalized_geometry in lands_gdf.normalized_geometry.tolist()],
            )

        # 4. Dilate and erode to tidy
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["buffered_geometry"] = pool.starmap(
                self.dilate_and_erode_polygon,
                [(buffered_geometry, 0.45) for buffered_geometry in lands_gdf.buffered_geometry.tolist()],
            )

        # 5. Remove empty geometries
        lands_gdf = lands_gdf[
            [not buffered_geometry.is_empty for buffered_geometry in lands_gdf.buffered_geometry.tolist()]
        ]

        # 6. Remove holes and multipolygon
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["buffered_geometry"] = pool.map(
                self.get_polygon_created_by_exterior, lands_gdf.buffered_geometry.tolist()
            )

        # 7. Simplify
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["simplified_geometry"] = pool.starmap(
                self.simplify_polygon,
                [
                    (buffered_geometry, self.SIMPLIFICATION_DEGREE)
                    for buffered_geometry in lands_gdf.buffered_geometry.tolist()
                ],
            )

        # 8. Remove invalid geometries
        lands_gdf = lands_gdf[
            [simplified_geometry.is_valid for simplified_geometry in lands_gdf.simplified_geometry.tolist()]
        ]

        # 9. Create centerlines of geometries
        lands_gdf["simplified_geometry_centerline"] = [
            self.get_polygon_centerline(simplified_geometry, self.TOLERANCE_CENTERLINE, min_branch_length=0.45)
            for simplified_geometry in lands_gdf.simplified_geometry.tolist()
        ]

        # 10. Remove empty centerlines
        lands_gdf = lands_gdf[
            [
                not simplified_geometry_centerline.is_empty
                for simplified_geometry_centerline in lands_gdf.simplified_geometry_centerline.tolist()
            ]
        ]

        # 11. Compute MRR(Minimum Rotated Rectangle) ratio
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["simplified_geometry_mrr_ratio"] = pool.map(
                self.compute_mrr_ratio, lands_gdf.simplified_geometry.tolist()
            )

        # 12. Compute degree sum of geometries
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            lands_gdf["simplified_geometry_degree_sum"] = pool.starmap(
                self.compute_polyon_degrees,
                [(simplified_geometry, True) for simplified_geometry in lands_gdf.simplified_geometry.tolist()],
            )

        # 13. Count axes of geometries from centerlines
        lands_gdf["axes_count"] = [
            max(0, len(simplified_geometry_centerline.coords) - 1)
            for simplified_geometry_centerline in lands_gdf.simplified_geometry_centerline.tolist()
        ]

        return lands_gdf

    def _get_reglar_lands(self, lands_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """_summary_

        Args:
            lands_gdf (gpd.GeoDataFrame): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """

        lands_gdf_regular = lands_gdf[
            (lands_gdf.iloc[:, 22] == self.SHAPE_HORIZONTAL_RECTANGLE)
            | (lands_gdf.iloc[:, 22] == self.SHAPE_VERTICAL_RECTANGLE)
            | (lands_gdf.iloc[:, 22] == self.SHAPE_LADDER)
            | (lands_gdf.iloc[:, 22] == self.SHAPE_SQUARE)
        ]

        lands_gdf_regular = lands_gdf_regular[
            lands_gdf_regular.apply(
                lambda row: row.simplified_geometry_mrr_ratio >= self.THRESHOLD_MRR_RATIO_REGULAR, axis=1
            )
        ]

        return lands_gdf_regular

    def _get_irregular_lands(self, lands_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """_summary_

        Args:
            lands_gdf (gpd.GeoDataFrame): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """

        lands_gdf_irregular = lands_gdf[
            (lands_gdf.iloc[:, 22] == self.SHAPE_IRREGULARITY)
            | (lands_gdf.iloc[:, 22] == self.SHAPE_FLAG)
            | (lands_gdf.iloc[:, 22] == self.SHAPE_UNDEFINED)
        ]

        th1 = self.THRESHOLD_MRR_RATIO_IRREGULAR_MAX
        th2 = self.THRESHOLD_MRR_RATIO_IRREGULAR_MIN
        th3 = self.THRESHOLD_INNDER_DEGREE_SUM_IRREGULAR

        lands_gdf_irregular = lands_gdf_irregular[
            (lands_gdf_irregular.apply(lambda row: th2 < row.simplified_geometry_mrr_ratio <= th1, axis=1))
            & (lands_gdf_irregular.apply(lambda row: row.simplified_geometry_degree_sum >= th3, axis=1))
            & (lands_gdf_irregular.apply(lambda row: 2 <= row.axes_count <= 3, axis=1))
        ]

        lands_gdf_irregular["simplified_geometry"] = lands_gdf_irregular.apply(
            lambda row: self.insert_intersected_vertices(row.simplified_geometry), axis=1
        )

        lands_gdf_irregular["splitters"] = lands_gdf_irregular.apply(
            lambda row: self.segment_polygon(
                polygon=row.simplified_geometry,
                number_to_split=row.axes_count,
                segment_threshold_length=self.SEGMENT_DIVIDE_BASELINE,
                even_area_weight=self.EVEN_AREA_WEIGHT,
                ombr_ratio_weight=self.OMBR_RATIO_WEIGHT,
                slope_similarity_weight=self.SLOPE_SIMILARITY_WEIGHT,
            )[-1],
            axis=1,
        )

        return lands_gdf_irregular

    def _visualize_geometries_as_grid(
        self, lands_gdf: gpd.GeoDataFrame, save_path: str, max_size_to_visualize: int = 1000
    ) -> None:
        """_summary_

        Args:
            geometries (_type_): _description_
            ncols (_type_): _description_
            figsize (tuple, optional): _description_. Defaults to (15, 15).
            save_path (str, optional): _description_. Defaults to "./".
            title (str, optional): _description_. Defaults to "".
        """

        def fig_to_img(figure):
            buf = io.BytesIO()
            figure.savefig(buf)
            buf.seek(0)
            image = Image.open(buf)

            return image

        dpi = 100
        figsize = (4, 4)
        col_num = 4
        row_num = int(np.ceil(min(lands_gdf.simplified_geometry.shape[0], max_size_to_visualize) / col_num))
        img_size = figsize[0] * dpi
        merged_image = Image.new("RGB", (col_num * img_size, row_num * img_size), "white")

        current_cols = 0
        output_height = 0
        output_width = 0
        color = "black"

        for ri, (loc, row) in enumerate(lands_gdf.iterrows()):
            if ri >= max_size_to_visualize:
                break

            figure = plt.figure(figsize=figsize, dpi=dpi)

            ax = figure.add_subplot(1, 1, 1)
            ax.axis("equal")

            ax.plot(*row.simplified_geometry.boundary.coords.xy, color=color, linewidth=0.6)
            ax.fill(*row.simplified_geometry.boundary.coords.xy, alpha=0.1, color=color)

            if row.get("splitters") is not None:
                for splitter in row.splitters:
                    ax.plot(*splitter.coords.xy, color="green", linewidth=0.6)

            x, y = row.simplified_geometry.boundary.coords.xy
            ax.scatter(x, y, color="red", s=7)

            ax.grid(True, color="lightgray")

            annotation = f"""
                iloc: {ri}
                loc: {loc}
                mrr_ratio: {row.simplified_geometry_mrr_ratio:.5f}
                degree_sum: {row.simplified_geometry_degree_sum:.5f}
                {save_path.split("processed")[-1]}
            """

            plt.axis([-2.0, 2.0, -2.0, 2.0])

            plt.gcf().text(
                0.45,
                0.2,
                annotation,
                va="center",
                ha="center",
                color="black",
                fontsize=8,
            )

            image = fig_to_img(figure)

            merged_image.paste(image, (output_width, output_height))

            current_cols += 1
            if current_cols >= col_num:
                output_width = 0
                output_height += img_size
                current_cols = 0
            else:
                output_width += img_size

            plt.close(figure)

        merged_image.save(save_path)

    def create(self) -> None:
        """Process and create data from shp files using geopandas.

        https://www.vworld.kr/dtna/dtna_fileDataView_s001.do

        Columns:
            A0    도형ID
            A1    고유번호
            A2    법정동코드
            A3    법정동명
            A4    대장구분코드
            A5    대장구분명
            A6    지번
            A7    지번지목부호
            A8    기준연도
            A9    기준월
            A10   지목코드
            A11   지목명
            A12   토지면적(㎡)
            A13   용도지역코드1
            A14   용도지역명1
            A15   용도지역코드2
            A16   용도지역명2
            A17   토지이용상황코드
            A18   토지이용상황명
            A19   지형높이코드
            A20   지형높이명
            A21   지형형상코드
            A22   지형형상명
            A23   도로측면코드
            A24   도로측면명
            A25   공시지가
            A26   데이터기준일자

            geometry 토지 도형

        """

        os.makedirs(self.LANDS_PATH, exist_ok=True)
        if len(os.listdir(self.LANDS_PATH)) < self.TOTAL_LANDS_FOLDER_COUNT:
            for file_name in os.listdir(self.LANDS_ZIP_PATH):
                file_path = os.path.join(self.LANDS_ZIP_PATH, file_name)

                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(self.LANDS_PATH)

        for _, folder in enumerate(os.listdir(self.LANDS_PATH)):
            if folder != "seoul-gangbukgu":
                continue

            folder_path = os.path.join(self.LANDS_PATH, folder)
            for shp_file in os.listdir(folder_path):
                if not shp_file.endswith(".shp"):
                    continue

                _lands_gdf = gpd.read_file(filename=os.path.join(folder_path, shp_file), encoding="CP949")

                lands_gdf = self._get_initial_lands_gdf(_lands_gdf)
                # lands_gdf_regular = self._get_reglar_lands(lands_gdf)
                lands_gdf_irregular = self._get_irregular_lands(lands_gdf)

                if self.is_debug_mode:
                    image_qa_path = os.path.join(self.IMG_QA_PATH, folder)
                    os.makedirs(image_qa_path, exist_ok=True)

                    # self._visualize_geometries_as_grid(
                    #     lands_gdf=lands_gdf_regular, save_path=os.path.join(image_qa_path, self.IMG_QA_NAME_REGULAR)
                    # )
                    self._visualize_geometries_as_grid(
                        lands_gdf=lands_gdf_irregular, save_path=os.path.join(image_qa_path, self.IMG_QA_NAME_IRREGULAR)
                    )

        # os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    data_creator = DataCreator(
        shp_dir=DataConfiguration.SHP_PATH,
        save_dir=DataConfiguration.SAVE_PATH,
        number_to_split=2,
        simplification_degree_factor=1.0,
        segment_threshold_length=5.0,
        is_debug_mode=True,
    )

    data_creator.create()
