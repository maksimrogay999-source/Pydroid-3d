import os, ctypes, sys
import sdl2, sdl2.ext
import numpy as np
import math
from PIL import Image
cimport numpy as cnp
from libcpp.vector cimport vector
from libc.stdio cimport fopen, fclose, FILE, fgets, sscanf
cimport cython
from libc.math cimport INFINITY,fabsf

# --- Настройки OpenGL для Pydroid3 -
os.environ['PYOPENGL_PLATFORM'] = 'egl'
try:
    ctypes.CDLL('libGLESv2.so', mode=ctypes.RTLD_GLOBAL)
except:
    pass
cimport gles2
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



cdef class GameObject:
    cdef public float x, y, z
    cdef public float scale_x, scale_y, scale_z
    cdef public float angle
    cdef public int vbo, count, texture_id
    cdef public float base_w, base_h, base_d

    def __init__(self, int vbo, int count, int texture_id, list base_size):
        self.vbo = vbo
        self.count = count
        self.texture_id = texture_id
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.angle = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0
        self.base_w = base_size[0]
        self.base_h = base_size[1]
        self.base_d = base_size[2]

    @property
    def size(self):
        return [self.base_w * self.scale_x, 
                self.base_h * self.scale_y, 
                self.base_d * self.scale_z]



# ---           КОЛЛИЗИИ            ---

cpdef bint is_collision(GameObject obj1, GameObject obj2):
    cdef float s1_w = obj1.base_w * obj1.scale_x
    cdef float s1_h = obj1.base_h * obj1.scale_y
    cdef float s1_d = obj1.base_d * obj1.scale_z
    
    cdef float s2_w = obj2.base_w * obj2.scale_x
    cdef float s2_h = obj2.base_h * obj2.scale_y
    cdef float s2_d = obj2.base_d * obj2.scale_z
    
    return (fabsf(obj1.x - obj2.x) * 2.0 < (s1_w + s2_w)) and \
           (fabsf(obj1.y - obj2.y) * 2.0 < (s1_h + s2_h)) and \
           (fabsf(obj1.z - obj2.z) * 2.0 < (s1_d + s2_d))


# ---           ДВИЖОК            ---

cdef class Engine:
    cdef object window
    cdef object context
    cdef object _cam
    cdef unsigned int default_tex
    
    cdef unsigned int shader
    cdef int u_proj
    cdef int u_view
    cdef int u_model
    

    cdef int width
    cdef int height

    def __init__(self, width=1080, height=1920):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("3D Engine v11", size=(width, height), flags=sdl2.SDL_WINDOW_OPENGL)
        self.window.show()
        self.context = sdl2.SDL_GL_CreateContext(self.window.window)
        gles2.glEnable(gles2.GL_DEPTH_TEST)
        
        self.width = width
        self.height = height
        self._cam = Camera()
        self.default_tex = self._create_white_texture()
        self._init_shaders()
        
        
        self.update_projection()

    cpdef update_projection(self):
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
        cdef float[::1] proj_view = proj 
        
        gles2.glUseProgram(self.shader)
        gles2.glUniformMatrix4fv(self.u_proj, 1, 0, &proj_view[0])

    def _create_white_texture(self):
        cdef cnp.uint8_t[:] data_view = np.array([255, 255, 255, 255], dtype=np.uint8)
        cdef unsigned int t_id
        gles2.glGenTextures(1,&t_id)
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, t_id)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MIN_FILTER, gles2.GL_LINEAR)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MAG_FILTER, gles2.GL_LINEAR)
        gles2.glTexImage2D(gles2.GL_TEXTURE_2D, 0, gles2.GL_RGBA, 1, 1, 0, gles2.GL_RGBA, gles2.GL_UNSIGNED_BYTE, &data_view[0])
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
        gles2.glUseProgram(self.shader)
        self.u_proj = glGetUniformLocation(self.shader, "proj")
        self.u_view = glGetUniformLocation(self.shader, "view")
        self.u_model = glGetUniformLocation(self.shader, "model")

    def load_texture(self, path):
        cdef unsigned int t_id
        cdef const unsigned char[::1] pixel_view
        if not os.path.exists(path): return self.default_tex
        try:
            img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
            gles2.glGenTextures(1,&t_id)
            gles2.glBindTexture(gles2.GL_TEXTURE_2D, t_id)
            # фильтрация для MAG_FILTER
            gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MIN_FILTER, gles2.GL_LINEAR)
            gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MAG_FILTER, gles2.GL_LINEAR)
            pixel_view = img.tobytes()
            gles2.glTexImage2D(gles2.GL_TEXTURE_2D, 0, gles2.GL_RGBA, img.width, img.height, 0, gles2.GL_RGBA, gles2.GL_UNSIGNED_BYTE, &pixel_view[0])
            return t_id
        except:
            return self.default_tex

    cpdef load_obj(self, filename, texture_id=None):
        cdef vector[vector[float]] v
        cdef vector[vector[float]] vt
        cdef vector[vector[float]] vn
        cdef vector[float] final_v
        
        cdef float min_p[3]
        cdef float max_p[3]
        for i in range(3):
            min_p[i] = INFINITY
            max_p[i] = -INFINITY


        cdef bytes fn_bytes = filename.encode('utf-8')
        cdef FILE* f = fopen(fn_bytes, "r")
        if f == NULL:
            print(f"Error: Could not open {filename}")
            return None

        cdef char line[512]
        cdef float t1, t2, t3
        cdef vector[float] tmp

        while fgets(line, 512, f):
            if line[0] == 'v':
                if line[1] == ' ':
                    sscanf(line, "v %f %f %f", &t1, &t2, &t3)
                    tmp = [t1, t2, t3]
                    v.push_back(tmp)
                    if t1 < min_p[0]: min_p[0] = t1
                    if t1 > max_p[0]: max_p[0] = t1
                    if t2 < min_p[1]: min_p[1] = t2
                    if t2 > max_p[1]: max_p[1] = t2
                    if t3 < min_p[2]: min_p[2] = t3
                    if t3 > max_p[2]: max_p[2] = t3
                elif line[1] == 't':
                    sscanf(line, "vt %f %f", &t1, &t2)
                    tmp = [t1, t2]
                    vt.push_back(tmp)
                elif line[1] == 'n':
                    sscanf(line, "vn %f %f %f", &t1, &t2, &t3)
                    tmp = [t1, t2, t3]
                    vn.push_back(tmp)

            elif line[0] == 'f' and line[1] == ' ':
                self._parse_face(line, v, vt, vn, final_v)

        fclose(f)


        cdef float[:] view = <float[:final_v.size()]>&final_v[0]
        arr = np.array(view, dtype=np.float32)


        size = [max_p[0] - min_p[0], max_p[1] - min_p[1], max_p[2] - min_p[2]]
        cdef unsigned int vbo
        gles2.glGenBuffers(1,&vbo)
        gles2.glBindBuffer(gles2.GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        

        return GameObject(vbo, len(final_v)//11, texture_id or self.default_tex, size)

    cdef void _parse_face(self, char* line, vector[vector[float]]& v, 
                         vector[vector[float]]& vt, vector[vector[float]]& vn, 
                         vector[float]& final_v):
        p_str = line.decode('utf-8').split()
        face_vertices = p_str[1:]
        
        if len(face_vertices) == 3:
            for v_str in face_vertices:
                self._add_vertex_to_final(v_str, v, vt, vn, final_v)
        elif len(face_vertices) == 4:
            v1_s, v2_s, v3_s, v4_s = face_vertices
            for s in [v1_s, v2_s, v3_s, v1_s, v3_s, v4_s]:
                self._add_vertex_to_final(s, v, vt, vn, final_v)

    cdef void _add_vertex_to_final(self, str v_str, vector[vector[float]]& v, 
                                 vector[vector[float]]& vt, vector[vector[float]]& vn, 
                                 vector[float]& final_v):
        parts = v_str.split('/')
        cdef int v_idx = int(parts[0]) - 1
        cdef int t_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
        cdef int n_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
        

        final_v.push_back(v[v_idx][0])
        final_v.push_back(v[v_idx][1])
        final_v.push_back(v[v_idx][2])
        final_v.push_back(1.0); final_v.push_back(1.0); final_v.push_back(1.0)
        if t_idx != -1:
            final_v.push_back(vt[t_idx][0]); final_v.push_back(vt[t_idx][1])
        else:
            final_v.push_back(0); final_v.push_back(0)
        if n_idx != -1:
            final_v.push_back(vn[n_idx][0]); final_v.push_back(vn[n_idx][1]); final_v.push_back(vn[n_idx][2])
        else:
            final_v.push_back(0); final_v.push_back(1.0); final_v.push_back(0)

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
        gles2.glUseProgram(self.shader)
        s, c = math.sin(obj.angle), math.cos(obj.angle)
        model = np.array([
            c * obj.scale_x, 0, s * obj.scale_x, 0,
            0, obj.scale_y, 0, 0,
            -s * obj.scale_z, 0, c * obj.scale_z, 0,
            obj.x, obj.y, obj.z, 1
        ], dtype=np.float32)
        cdef float[:] model_view = model.view(np.float32).flatten()
        gles2.glUniformMatrix4fv(self.u_model, 1, gles2.GL_FALSE, &model_view[0])
        self.update_projection() 
        matrix = self._cam.get_view_matrix()
        cdef float[:] matrix_view = matrix.view(np.float32).flatten()
        gles2.glUniformMatrix4fv(self.u_view, 1, gles2.GL_FALSE, &matrix_view[0])
        
        
        gles2.glBindBuffer(gles2.GL_ARRAY_BUFFER, obj.vbo)
        for name, size, offset in [("pos",3,0), ("col",3,12), ("uv",2,24), ("norm",3,32)]:
            loc = glGetAttribLocation(self.shader, name)
            if loc != -1:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(offset))
        
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, obj.texture_id)
        glDrawArrays(GL_TRIANGLES, 0, obj.count)