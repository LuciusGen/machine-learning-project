import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy import ndarray
from rembg import remove
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing


@dataclass
class ContourWithArea:
    """
    Object for contour and area data
    """
    contour: np.array
    area: float


@dataclass
class WhiteListRealSize:
    """
    Real white list size in mm
    """
    width = 210
    height = 297
    area = width * height


class ObjectsWorker:
    """
    Object to get real object contours and areas
    """
    def get_real_items_contour_with_area(self, image) -> ndarray[ContourWithArea]:
        contours = self.__get_object_contours(image)
        k = self.__get_koef_for_transform(image)

        return self.__filter_contour([ContourWithArea((i / k).astype(int), cv2.contourArea((i / k).astype(int))) for i in contours])

    def __get_object_contours(self, image) -> list[ndarray]:
        only_objects = remove(image)
        gray = cv2.cvtColor(only_objects, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 1, 255, 0)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_contours = self.__get_sorted_contours(contours)

        return sorted_contours

    def __get_koef_for_transform(self, img) -> float:
        real_size = WhiteListRealSize()
        white_area = self.__get_white_list_area(img)
        k = math.sqrt(white_area / real_size.area)

        return k

    def __filter_contour(self, contours_with_areas: list[ContourWithArea]) -> ndarray[ContourWithArea]:
        filter_area = 50
        filtered_arr = [i.area > filter_area for i in contours_with_areas]
        ct = np.array(contours_with_areas)
        return ct[filtered_arr]

    def __get_white_list_area(self, img) -> float:
        img_gray = rgb2gray(img)

        thresh_otsu = threshold_otsu(img_gray)
        otsu = img_gray >= thresh_otsu

        close_otsu = binary_closing(otsu)
        cv_close_otsu = img_as_ubyte(close_otsu)

        contours = cv2.findContours(cv_close_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        cnt = self.__get_sorted_contours(contours)[0]

        rbox = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rbox).astype(np.int32)

        return cv2.contourArea(pts)

    @staticmethod
    def __get_sorted_contours(contours) -> list[ndarray]:
        return sorted(contours, key=cv2.contourArea, reverse=True)
