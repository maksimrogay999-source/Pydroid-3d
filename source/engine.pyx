import os, ctypes, sys
import numpy as np
import math,time
from PIL import Image
from libcpp.vector cimport vector
from libc.stdio cimport fopen, fclose, FILE, fgets, sscanf
from libc.stdint cimport uintptr_t
from libc.math cimport INFINITY,fabsf
cimport gles2
cimport sdl
cimport cython
cimport numpy as cnp

cpdef unsigned int load_shaders(str v_code, str f_code):
    v_bytes = v_code.encode('utf-8')
    f_bytes = f_code.encode('utf-8')
    cdef const char* vs_c = v_bytes
    cdef const char* fs_c = f_bytes
    cdef unsigned int vs, fs, prog
    vs = gles2.glCreateShader(gles2.GL_VERTEX_SHADER)
    gles2.glShaderSource(vs, 1, &vs_c, NULL)
    gles2.glCompileShader(vs)
    fs = gles2.glCreateShader(gles2.GL_FRAGMENT_SHADER)
    gles2.glShaderSource(fs, 1, &fs_c, NULL)
    gles2.glCompileShader(fs)
    prog = gles2.glCreateProgram()
    gles2.glAttachShader(prog, vs)
    gles2.glAttachShader(prog, fs)
    gles2.glLinkProgram(prog)
    
    return prog

from libc.stdlib cimport malloc, free

cdef class FBO:
    cdef public unsigned int id
    cdef public unsigned int texture
    cdef public unsigned int rbo
    cdef int width, height

    def __init__(self, int width, int height):
        self.width = width
        self.height = height
        gles2.glGenTextures(1, &self.texture)
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, self.texture)
        gles2.glTexImage2D(gles2.GL_TEXTURE_2D, 0, gles2.GL_RGBA, 
                           width, height, 0, gles2.GL_RGBA, 
                           gles2.GL_UNSIGNED_BYTE, NULL)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MIN_FILTER, gles2.GL_LINEAR)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MAG_FILTER, gles2.GL_LINEAR)
        gles2.glGenFramebuffers(1, &self.id)
        gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, self.id)
        gles2.glFramebufferTexture2D(gles2.GL_FRAMEBUFFER, gles2.GL_COLOR_ATTACHMENT0, 
                                     gles2.GL_TEXTURE_2D, self.texture, 0)
        gles2.glGenRenderbuffers(1, &self.rbo)
        gles2.glBindRenderbuffer(gles2.GL_RENDERBUFFER, self.rbo)
        gles2.glRenderbufferStorage(gles2.GL_RENDERBUFFER, gles2.GL_DEPTH_COMPONENT16, width, height)
        gles2.glFramebufferRenderbuffer(gles2.GL_FRAMEBUFFER, gles2.GL_DEPTH_ATTACHMENT, 
                                        gles2.GL_RENDERBUFFER, self.rbo)

        if gles2.glCheckFramebufferStatus(gles2.GL_FRAMEBUFFER) != gles2.GL_FRAMEBUFFER_COMPLETE:
            print("Ошибка: FBO не укомплектован!")

        gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, 0)


    cpdef bind(self):
        gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, self.id)

        gles2.glViewport(0, 0, self.width, self.height)

    cpdef unbind(self, int screen_w, int screen_h):

        gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, 0)

        gles2.glViewport(0, 0, screen_w, screen_h)

    def __dealloc__(self):
        gles2.glDeleteFramebuffers(1, &self.id)
    cpdef save_screenshot(self, str filename):
        cdef int size = self.width * self.height * 4
        cdef unsigned char* data = <unsigned char*>malloc(size)
    
        try:
            gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, self.id)
            gles2.glReadPixels(0, 0, self.width, self.height, gles2.GL_RGBA, gles2.GL_UNSIGNED_BYTE, data)
            img_bytes = (<char*>data)[:size]
            img = Image.frombytes("RGBA", (self.width, self.height), img_bytes)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(filename)
            print(f"Скриншот сохранен: {filename}")
        
        finally:
            free(data)
            gles2.glBindFramebuffer(gles2.GL_FRAMEBUFFER, 0)


class Camera:
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 5.0
        self.pitch = 0.0
        self.yaw = -math.pi/2
        self.size = [0.8, 1.8, 0.8]

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
cdef class Sun:
    cdef public double x,y,z
    def __init__(self):
        self.x = 1.0
        self.y= 2.0
        self.z = 1.0

cdef class Engine:
    cdef sdl.SDL_Window* window
    cdef sdl.SDL_GLContext context
    cdef object _cam
    cdef unsigned int default_tex
    cdef public object sun
    
    cdef unsigned int shader
    cdef unsigned int gui_shader
    cdef int u_proj
    cdef int u_view
    cdef int u_model
    cdef int u_lightpos
    

    cdef int width
    cdef int height
    
    cdef long last
    cdef double dt

    def __init__(self, width=1080, height=1920):
        self.dt = 0
        self.last = time.perf_counter_ns()
        self.sun = Sun()
        sdl.SDL_Init(sdl.SDL_INIT_EVERYTHING)
        self.window = sdl.SDL_CreateWindow("3D Engine v11", 0,0,width, height, sdl.SDL_WINDOW_OPENGL)
        sdl.SDL_ShowWindow(self.window)
        self.context = sdl.SDL_GL_CreateContext(self.window)
        gles2.glEnable(gles2.GL_DEPTH_TEST)
        gles2.glEnable(gles2.GL_BLEND)
        gles2.glBlendFunc(gles2.GL_SRC_ALPHA, gles2.GL_ONE_MINUS_SRC_ALPHA)
        
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

    cpdef _create_white_texture(self):
        cdef cnp.uint8_t[:] data_view = np.array([255, 255, 255, 255], dtype=np.uint8)
        cdef unsigned int t_id
        gles2.glGenTextures(1,&t_id)
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, t_id)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MIN_FILTER, gles2.GL_LINEAR)
        gles2.glTexParameteri(gles2.GL_TEXTURE_2D, gles2.GL_TEXTURE_MAG_FILTER, gles2.GL_LINEAR)
        gles2.glTexImage2D(gles2.GL_TEXTURE_2D, 0, gles2.GL_RGBA, 1, 1, 0, gles2.GL_RGBA, gles2.GL_UNSIGNED_BYTE, &data_view[0])
        return t_id

    cpdef camera(self): return self._cam

    cpdef _init_shaders(self):
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
        uniform vec3 u_lightPos;
        void main() {
            vec3 light = normalize(vec3(u_lightPos));
            float diff = max(dot(normalize(v_norm), light), 0.25);
            gl_FragColor = texture2D(tex, v_uv) * vec4(v_col * diff, 1.0);
        }"""
        self.shader = load_shaders(v_s,f_s)
        gles2.glUseProgram(self.shader)
        self.u_proj = gles2.glGetUniformLocation(self.shader, "proj")
        self.u_view = gles2.glGetUniformLocation(self.shader, "view")
        self.u_model = gles2.glGetUniformLocation(self.shader, "model")
        self.u_lightpos = gles2.glGetUniformLocation(self.shader, "u_lightPos")
        cdef str v_code = """
attribute vec2 position;
attribute vec2 texCoord;
varying vec2 v_uv;
uniform mat4 u_proj;
uniform vec4 u_rect;

void main() {
    v_uv = texCoord;
    vec2 pos = position * u_rect.zw + u_rect.xy;
    gl_Position = u_proj * vec4(pos, 0.0, 1.0);
}

        """
        cdef str f_code = """
        precision mediump float;
        varying vec2 v_uv;
        uniform sampler2D u_texture;
        void main() {
            gl_FragColor = texture2D(u_texture, v_uv);
        }
        """

        self.gui_shader = load_shaders(v_code, f_code)

    cpdef load_texture(self, path):
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
        size = [max_p[0] - min_p[0], max_p[1] - min_p[1], max_p[2] - min_p[2]]


        cdef unsigned int vbo
        gles2.glGenBuffers(1, &vbo)
        gles2.glBindBuffer(gles2.GL_ARRAY_BUFFER, vbo)
        cdef size_t buffer_size = final_v.size() * sizeof(float)
        gles2.glBufferData(
            gles2.GL_ARRAY_BUFFER, 
            buffer_size, 
            &final_v[0], 
            gles2.GL_STATIC_DRAW
        )

        

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
    cpdef get_dt(self):
        return self.dt
    cpdef wait(self,sec):
        sdl.SDL_Delay(sec*1000)
    cpdef main(self):
         sdl.SDL_GL_SwapWindow(self.window)
         cdef long time_c = time.perf_counter_ns()
         self.dt = (time_c - self.last) / 1e9
         self.last = time_c
    cpdef ScreenClear(self):
        gles2.glClear(gles2.GL_COLOR_BUFFER_BIT | gles2.GL_DEPTH_BUFFER_BIT)
    cpdef ScreenColor(self, float r, float g, float b, float a=1.0):
        gles2.glClearColor(r, g, b, a)

    cpdef draw(self, obj):
        if not obj: return
        gles2.glEnable(gles2.GL_DEPTH_TEST)
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
        gles2.glUniform3f(self.u_lightpos, self.sun.x, self.sun.y, self.sun.z)
        
        gles2.glBindBuffer(gles2.GL_ARRAY_BUFFER, obj.vbo)
        for name, size, offset in [("pos",3,0), ("col",3,12), ("uv",2,24), ("norm",3,32)]:
            name_bytes = name.encode('utf-8') 
            loc = gles2.glGetAttribLocation(self.shader, name_bytes)
            if loc != -1:
                gles2.glEnableVertexAttribArray(loc)
                gles2.glVertexAttribPointer(loc, size, gles2.GL_FLOAT, gles2.GL_FALSE, 44, <const void*><uintptr_t>offset)
        
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, obj.texture_id)
        gles2.glDrawArrays(gles2.GL_TRIANGLES, 0, obj.count)

    cpdef draw_gui(self, unsigned int texture_id, float x, float y, float w, float h):
        cdef float sw = <float>self.width
        cdef float sh = <float>self.height
        cdef float[:] view = np.array([
            0.0, 0.0,  0.0, 0.0, 
            1.0, 0.0,  1.0, 0.0, 
            0.0, 1.0,  0.0, 1.0, 
            1.0, 1.0,  1.0, 1.0  
        ], dtype=np.float32)

        gles2.glUseProgram(self.gui_shader)
        gles2.glDisable(gles2.GL_DEPTH_TEST)
        gles2.glBindBuffer(gles2.GL_ARRAY_BUFFER, 0)
        cdef int rect_loc = gles2.glGetUniformLocation(self.gui_shader, "u_rect")
        gles2.glUniform4f(rect_loc, x, y, w, h)
        cdef float proj[16]
        for i in range(16): proj[i] = 0.0
        proj[0] = 2.0 / sw;    proj[12] = -1.0
        proj[5] = -2.0 / sh;   proj[13] = 1.0
        proj[10] = 1.0;        proj[15] = 1.0

        cdef int proj_loc = gles2.glGetUniformLocation(self.gui_shader, "u_proj")
        gles2.glUniformMatrix4fv(proj_loc, 1, 0, proj)
        gles2.glActiveTexture(0x84C0)
        gles2.glBindTexture(gles2.GL_TEXTURE_2D, texture_id)
        gles2.glUniform1i(gles2.glGetUniformLocation(self.gui_shader, "u_texture"), 0)

        cdef int pos_loc = gles2.glGetAttribLocation(self.gui_shader, "position")
        cdef int uv_loc = gles2.glGetAttribLocation(self.gui_shader, "texCoord")
        
        gles2.glEnableVertexAttribArray(pos_loc)
        gles2.glVertexAttribPointer(pos_loc, 2, 0x1406, 0, 16, &view[0])
        gles2.glEnableVertexAttribArray(uv_loc)
        gles2.glVertexAttribPointer(uv_loc, 2, 0x1406, 0, 16, &view[2])

        gles2.glDrawArrays(0x0005, 0, 4)
        gles2.glEnable(gles2.GL_DEPTH_TEST)

