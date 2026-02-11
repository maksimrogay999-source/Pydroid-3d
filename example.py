import sdl2 # Костыль включающий графический режим
from main import *
if __name__ == "__main__":
    core = Engine()

    cam = core.camera()
        
    tex = core.load_texture("tex.png")
    box = core.load_obj("box.obj", tex)

    running = True
    while running:
        core.ScreenColor(0.1, 0.1, 0.15, 1.0)
        core.ScreenClear()
        if box:
            box.angle += 0.02
            box.z -= 0.02
            core.draw(box)

        core.main()
        core.wait(16)

