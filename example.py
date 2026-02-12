import sdl2 # Костыль включающий графический режим
import random
from engine_core import *
if __name__ == "__main__":
    core = Engine()

    cam = core.camera()
        
    tex = core.load_texture("tex.png")
    two = core.load_texture("tex2.jpg")
    boxs = []
    for i in range(100):
    	b = core.load_obj("box.obj", random.choice([tex, two]))
    	b.x = random.uniform(-5, 5)
    	b.y = random.uniform(-5, 5)
    	b.z = random.uniform(-10, -2)
    	boxs.append(b)

    running = True
    while running:
        core.ScreenColor(0.1, 0.1, 0.15, 1.0)
        core.ScreenClear()
        for box in boxs:
        	if box:
        	   	box.angle += 0.01
        	   	box.z-=0.02
        	   	core.draw(box)

        core.main()