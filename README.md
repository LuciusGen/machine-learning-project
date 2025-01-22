# Постановка задачи Intelligent Placer
Требуется создать “Intelligent Placer”: по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны. Также заранее известно направление вертикальной оси Z у этих предметов. Многоугольник задается массивом с координатами вершин на естественной плоскости.

## Оформление проекта
 “Intelligent Placer” должен быть оформлен в виде python-библиотеки intelligent_placer_lib, которая поставляется каталогом intelligent_placer_lib с файлом intelligent_placer.py, содержащим функцию - точку входа 
 
def check_image (<path_to_png_jpg_image_on_local_computer>[, <poligon_coordinates>])

Возвращает True если предметы могут влезть в многоугольник, иначе False.

## Требования к изображеиям:

1) Объекты на изображении должны быть хорошо различимы(выделяться на фоне белого листа).
2) Лист, на котором находяться объекты, должен быть форматом А4.
3) Предметы не пересекаются с краем фонового листа.
4) Углы и края белого листа и объекта попадают на фото целиком не перекрываясь.
5) Камера должна смотреть на объект вертикально или под небольшим углом.
6) Изображения не должны быть сильно зашумлены.

## Многоугольник
* Многоугольник задается массивом координат вершин на естественной плоскости.
* В качестве единицы в системе коорднат выступают миллиметры. 
* Многоугольник должен являться замкнутой фигурой.
* Массив координат должен быть упорядочен(каждая предыдущая вершина будет соединяться со следующей).

