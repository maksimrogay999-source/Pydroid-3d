# Pydroid-3D

Лёгкий 3D-движок для Android (Pydroid3) на OpenGL ES 2.0.

## Возможности
- Загрузка OBJ с текстурами и нормалями
- Камера от 1-го лица (pitch/yaw)
- Коллизии (AABB)
- Текстуры (PNG через Pillow)

## Установка в Pydroid3
1. Установите пакеты: `pip install pysdl2 pillow numpy PyOpenGL`
2. Скопируйте репозиторий
3. Установите модуль: `python build.py`
4. Запустите `example.py`

## Пример
```python
from engine_core import *
core = Engine()
box = core.load_obj("box.obj")
while True:
    box.angle += 0.02
    core.draw(box)
    core.main()
```
![Скриншот работы движка](screenshots/demo.jpg)