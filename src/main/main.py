import cv2
from shapely.geometry import Polygon

from src.main.ObjectsWorker import ObjectsWorker
from src.main.Placer import Placer


def check_image(path: str, contour: Polygon):
    objects_worker = ObjectsWorker()
    placer = Placer()

    img = cv2.imread(path)

    items_contours_with_areas = objects_worker.get_real_items_contour_with_area(img)
    return placer.place_objects(contour, items_contours_with_areas)


if __name__ == '__main__':
    polygon = Polygon([[0, 0], [60, 0], [60, 90], [0, 90]])
    print(check_image("E:\\teach\\2023\\machine-learning-project-tmp\\resources\\testData\\more\\usb.jpg", polygon))

