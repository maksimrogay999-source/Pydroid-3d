import os, ctypes, sys
import sdl2, sdl2.ext
import numpy as np
import math
from PIL import Image

# --- Настройки OpenGL для Pydroid3 -
os.environ['PYOPENGL_PLATFORM'] = 'egl'
try:
    ctypes.CDLL('libGLESv2.so', mode=ctypes.RTLD_GLOBAL)
except:
    pass

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class Camera:
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 5.0
        self.pitch = 0.0 # Наклон головы
        self.yaw = -math.pi/2 # Поворот головы
        self.size = [0.8, 1.8, 0.8] # Хитбокс игрока

    def get_view_matrix(self):
        cos_p = math.cos(self.pitch); sin_p = math.sin(self.pitch)
        cos_y = math.cos(self.yaw); sin_y = math.sin(self.yaw)
        forward = np.array([cos_p * cos_y, sin_p, cos_p * sin_y])
        up = np.array([0, 1, 0])
        zaxis = -forward / np.linalg.norm(forward)
        xaxis = np.cross(up, zaxis); xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        
        view = np.identity(4, dtype=np.float32)
        view[0, :3] = xaxis; view[1, :3] = yaxis; view[2, :3] = zaxis
        view[:3, 3] = [-np.dot(xaxis, [self.x, self.y, self.z]),
                       -np.dot(yaxis, [self.x, self.y, self.z]),
                       -np.dot(zaxis, [self.x, self.y, self.z])]
        return view.T

class GameObject:
    def __init__(self, vbo, count, texture_id, base_size):
        self.vbo = vbo
        self.count = count
        self.texture_id = texture_id
        self.x = self.y = self.z = 0.0
        self.angle = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0
        self.base_size = base_size

# ---           КОЛЛИЗИИ            ---

def is_collision(obj1, obj2):
    s1 = obj1.size if hasattr(obj1, 'size') else [obj1.base_size[0]*obj1.scale_x, obj1.base_size[1]*obj1.scale_y, obj1.base_size[2]*obj1.scale_z]
    s2 = obj2.size if hasattr(obj2, 'size') else [obj2.base_size[0]*obj2.scale_x, obj2.base_size[1]*obj2.scale_y, obj2.base_size[2]*obj2.scale_z]
    
    return (abs(obj1.x - obj2.x) * 2 < (s1[0] + s2[0])) and \
           (abs(obj1.y - obj2.y) * 2 < (s1[1] + s2[1])) and \
           (abs(obj1.z - obj2.z) * 2 < (s1[2] + s2[2]))

# ---           ДВИЖОК            ---

class Engine:
    def __init__(self, width=1080, height=1920):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("3D Engine v11", size=(width, height), flags=sdl2.SDL_WINDOW_OPENGL)
        self.window.show()
        self.context = sdl2.SDL_GL_CreateContext(self.window.window)
        glEnable(GL_DEPTH_TEST)
        
        self.width = width
        self.height = height
        self._cam = Camera()
        self.default_tex = self._create_white_texture()
        self._init_shaders()
        
        
        self.update_projection()

    def update_projection(self):
        aspect = self.width / self.height
        fov = math.radians(45)
        f = 1.0 / math.tan(fov / 2.0)
        near, far = 0.1, 100.0
        # Матрица перспективы
        proj = np.array([
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), -1,
            0, 0, (2*far*near)/(near-far), 0
        ], dtype=np.float32)
        
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.u_proj, 1, GL_FALSE, proj)

    def _create_white_texture(self):
        data = np.array([255, 255, 255, 255], dtype=np.uint8)
        t_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, t_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return t_id

    def camera(self): return self._cam

    def _init_shaders(self):
        v_s = """
        attribute vec3 pos; attribute vec3 col; attribute vec2 uv; attribute vec3 norm;
        varying vec3 v_col; varying vec2 v_uv; varying vec3 v_norm;
        uniform mat4 proj, view, model;
        void main() {
            v_col = col; v_uv = uv; v_norm = mat3(model) * norm;
            gl_Position = proj * view * model * vec4(pos, 1.0);
        }"""
        f_s = """
        precision mediump float; varying vec3 v_col; varying vec2 v_uv; varying vec3 v_norm;
        uniform sampler2D tex;
        void main() {
            vec3 light = normalize(vec3(1.0, 2.0, 1.0));
            float diff = max(dot(normalize(v_norm), light), 0.25);
            gl_FragColor = texture2D(tex, v_uv) * vec4(v_col * diff, 1.0);
        }"""
        self.shader = compileProgram(compileShader(v_s, GL_VERTEX_SHADER), compileShader(f_s, GL_FRAGMENT_SHADER))
        glUseProgram(self.shader)
        self.u_proj = glGetUniformLocation(self.shader, "proj")
        self.u_view = glGetUniformLocation(self.shader, "view")
        self.u_model = glGetUniformLocation(self.shader, "model")

    def load_texture(self, path):
        if not os.path.exists(path): return self.default_tex
        try:
            img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
            t_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, t_id)
            # фильтрация для MAG_FILTER
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
            return t_id
        except:
            return self.default_tex

    def load_obj(self, filename, texture_id=None):
        v, vt, vn, final_v = [], [], [], []
        min_p = [float('inf')]*3; max_p = [float('-inf')]*3
        try:
            with open(filename, 'r') as f:
                for line in f:
                    p = line.split()
                    if not p: continue
                    if p[0] == 'v':
                        pts = list(map(float, p[1:4]))
                        v.append(pts)
                        for i in range(3):
                            min_p[i] = min(min_p[i], pts[i]); max_p[i] = max(max_p[i], pts[i])
                    elif p[0] == 'vt': vt.append(list(map(float, p[1:3])))
                    elif p[0] == 'vn': vn.append(list(map(float, p[1:4])))
                    elif p[0] == 'f':
                        for vert in p[1:4]:
                            parts = vert.split('/')
                            v_idx = int(parts[0]) - 1
                            t_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                            n_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
                            
                            p_d = v[v_idx]
                            u_d = vt[t_idx] if t_idx != -1 else [0, 0]
                            n_d = vn[n_idx] if n_idx != -1 else [0, 1, 0]
                            
                            # 3(pos) + 3(col) + 2(uv) + 3(norm) = 11 float
                            final_v.extend([p_d[0], p_d[1], p_d[2], 1, 1, 1, u_d[0], u_d[1], n_d[0], n_d[1], n_d[2]])
            
            size = [max_p[i] - min_p[i] for i in range(3)]
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            arr = np.array(final_v, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
            return GameObject(vbo, len(final_v)//11, texture_id or self.default_tex, size)
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return None
    def wait(self,sec):
    	sdl2.SDL_Delay(sec)
    def main(self):
    	 sdl2.SDL_GL_SwapWindow(self.window.window)
    def ScreenClear(self):
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    def ScreenColor(self,*args):
    	glClearColor(*args)
    def draw(self, obj):
        if not obj: return
        glUseProgram(self.shader)
        s, c = math.sin(obj.angle), math.cos(obj.angle)
        model = np.array([
            c * obj.scale_x, 0, s * obj.scale_x, 0,
            0, obj.scale_y, 0, 0,
            -s * obj.scale_z, 0, c * obj.scale_z, 0,
            obj.x, obj.y, obj.z, 1
        ], dtype=np.float32)
        
        glUniformMatrix4fv(self.u_model, 1, GL_FALSE, model)
        self.update_projection() 
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, self._cam.get_view_matrix())
        
        
        glBindBuffer(GL_ARRAY_BUFFER, obj.vbo)
        for name, size, offset in [("pos",3,0), ("col",3,12), ("uv",2,24), ("norm",3,32)]:
            loc = glGetAttribLocation(self.shader, name)
            if loc != -1:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(offset))
        
        glBindTexture(GL_TEXTURE_2D, obj.texture_id)
        glDrawArrays(GL_TRIANGLES, 0, obj.count)