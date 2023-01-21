import numpy as np
from scipy.optimize import minimize
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
from ObjectsWorker import ContourWithArea


class Placer:
    """
    Object that recognize can we place objects in area or not
        1) compare areas
        2) try to place objects in contour
    """

    def place_objects(self, contour: Polygon, items_contour_with_area: list[ContourWithArea]) -> bool:
        if self.__compare_areas(contour.area, items_contour_with_area):
            return False

        contour_polygon = Polygon(contour)
        items_polygon = self.__convert_items_contours_to_polygon(items_contour_with_area)

        return self.__optimize(contour_polygon, items_polygon)

    def __optimize(self, contour_polygon: Polygon, items_polygon: list[Polygon]) -> bool:
        plt.plot(*contour_polygon.exterior.xy)
        res_contour_polygon = Polygon(contour_polygon)
        for item_polygon in items_polygon:
            centered_item_polygon = self.__shift_to_polygon_center(res_contour_polygon, item_polygon)
            angle, x, y = self.__optimize_iteration(res_contour_polygon, centered_item_polygon).x
            shifted_polygon = translate(centered_item_polygon, x, y)
            rotated_shifted_polygon = rotate(shifted_polygon, angle)

            if not self.__check_condition(res_contour_polygon, rotated_shifted_polygon):
                plt.plot(*rotated_shifted_polygon.exterior.xy)
                plt.show()
                return False

            res_contour_polygon = res_contour_polygon - rotated_shifted_polygon

            plt.plot(*rotated_shifted_polygon.exterior.xy)

        plt.show()
        return True

    def __optimize_iteration(self, contour: Polygon, item_polygon: Polygon):
        def optimize_function(polygon_optimization_params: tuple[float, float, float]):
            """
            optimize item place
            :param polygon_optimization_params: angle, x_shift, y_shift
            :return: cell function value
            """
            angle, x, y = polygon_optimization_params
            shifted_polygon = translate(item_polygon, x, y)
            rotated_shifted_polygon = rotate(shifted_polygon, angle)
            return -contour.intersection(rotated_shifted_polygon).area - contour.centroid.distance(
                shifted_polygon.centroid)

        contour_min_x, contour_min_y, contour_max_x, contour_max_y = contour.bounds
        item_polygon_min_x, item_polygon_min_y, item_polygon_max_x, item_polygon_max_y = item_polygon.bounds

        shift_x = ((contour_max_x - contour_min_x) - min(item_polygon_max_x - item_polygon_min_x,
                                                         item_polygon_max_y - item_polygon_min_y)) / 2
        shift_y = ((contour_max_y - contour_min_y) - min(item_polygon_max_x - item_polygon_min_x,
                                                         item_polygon_max_y - item_polygon_min_y)) / 2

        return minimize(optimize_function,
                        x0=np.array([0, 0, 0]), method='trust-constr',
                        bounds=((0, 360), (-shift_x, shift_x), (-shift_y, shift_y))
                        )

    def __check_condition(self, contour: Polygon, item_polygon: Polygon):
        # contour_min_x, contour_min_y, contour_max_x, contour_max_y = contour.bounds
        # item_polygon_min_x, item_polygon_min_y, item_polygon_max_x, item_polygon_max_y = item_polygon.bounds
        # return max(contour_max_x - contour_min_x, contour_max_y, contour_min_y) > \
        #        max(item_polygon_max_x - item_polygon_min_x, item_polygon_max_y - item_polygon_min_y)
        item_area = item_polygon.area
        item_in_contour_area = contour.intersection(item_polygon).area

        return item_area * 1.05 >= item_in_contour_area >= item_area * .95

    def __convert_items_contours_to_polygon(self, items_contour_with_area: list[ContourWithArea]):
        return [Polygon(np.squeeze(i.contour)) for i in items_contour_with_area]

    def __compare_areas(self, contour_area: float, items_contour_with_area: list[ContourWithArea]) -> bool:
        sum_items_area = sum([i.area for i in items_contour_with_area])

        return sum_items_area > contour_area

    def __shift_to_polygon_center(self, contour: Polygon, item_contour: Polygon) -> Polygon:
        center_contour_x, center_contour_y = contour.centroid.coords.xy
        center_item_contour_x, center_item_contour_y = item_contour.centroid.coords.xy

        center_shift_x = center_contour_x[0] - center_item_contour_x[0]
        center_shift_y = center_contour_y[0] - center_item_contour_y[0]

        return translate(item_contour, center_shift_x, center_shift_y)
