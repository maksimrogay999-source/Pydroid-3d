import os, ctypes, sys
import numpy as np
import math,time
from PIL import Image
from libcpp.vector cimport vector
from libc.stdio cimport fopen, fclose, FILE, fgets, sscanf
from libc.stdint cimport uintptr_t
from libc.math cimport INFINITY,fabsf
import glRender as render
ffi = render.ffi
cimport sdl
cimport cython
cimport numpy as cnp
from libc.stdlib cimport malloc, free


cdef class FBO:
    cdef public unsigned int id
    cdef public unsigned int texture
    cdef public unsigned int rbo
    cdef int width, height

    def __init__(self, int width, int height):
        self.width = width
        self.height = height
        render.glGenTextures(1, ffi.cast("GLuint *",<uintptr_t>&self.texture))
        render.glBindTexture(render.GL_TEXTURE_2D, self.texture)
        render.glTexImage2D(render.GL_TEXTURE_2D, 0, render.GL_RGBA, 
                           width, height, 0, render.GL_RGBA, 
                           render.GL_UNSIGNED_BYTE, ffi.NULL)
        render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MIN_FILTER, render.GL_LINEAR)
        render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MAG_FILTER, render.GL_LINEAR)
        render.glGenFramebuffers(1, ffi.cast("GLuint *",<uintptr_t>&self.id))
        render.glBindFramebuffer(render.GL_FRAMEBUFFER, self.id)
        render.glFramebufferTexture2D(render.GL_FRAMEBUFFER, render.GL_COLOR_ATTACHMENT0, 
                                     render.GL_TEXTURE_2D, self.texture, 0)
        render.glGenRenderbuffers(1, ffi.cast("GLuint *", <uintptr_t>&self.rbo))
        render.glBindRenderbuffer(render.GL_RENDERBUFFER, self.rbo)
        render.glRenderbufferStorage(render.GL_RENDERBUFFER, render.GL_DEPTH_COMPONENT16, width, height)
        render.glFramebufferRenderbuffer(render.GL_FRAMEBUFFER, render.GL_DEPTH_ATTACHMENT, 
                                        render.GL_RENDERBUFFER, self.rbo)

        if render.glCheckFramebufferStatus(render.GL_FRAMEBUFFER) != render.GL_FRAMEBUFFER_COMPLETE:
            print("Ошибка: FBO не укомплектован!")

        render.glBindFramebuffer(render.GL_FRAMEBUFFER, 0)


    cpdef bind(self):
        render.glBindFramebuffer(render.GL_FRAMEBUFFER, self.id)

        render.glViewport(0, 0, self.width, self.height)

    cpdef unbind(self, int screen_w, int screen_h):

        render.glBindFramebuffer(render.GL_FRAMEBUFFER, 0)

        render.glViewport(0, 0, screen_w, screen_h)

    def __dealloc__(self):
        render.glDeleteFramebuffers(1, ffi.cast("GLuint *",<uintptr_t>&self.id))
    cpdef save_screenshot(self, str filename):
        cdef int size = self.width * self.height * 4
        cdef unsigned char* data = <unsigned char*>malloc(size)
    
        try:
            render.glBindFramebuffer(render.GL_FRAMEBUFFER, self.id)
            render.glReadPixels(0, 0, self.width, self.height, render.GL_RGBA, render.GL_UNSIGNED_BYTE, data)
            img_bytes = (<char*>data)[:size]
            img = Image.frombytes("RGBA", (self.width, self.height), img_bytes)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(filename)
            print(f"Скриншот сохранен: {filename}")
        
        finally:
            free(data)
            render.glBindFramebuffer(render.GL_FRAMEBUFFER, 0)


from libc.math cimport cos, sin, sqrt

cdef class Camera:
    cdef public double x, y, z, pitch, yaw
    cdef public float size[3]
    
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 5.0
        self.pitch = 0.0
        self.yaw = -3.141592653589793 / 2 
        self.size[0] = 0.8
        self.size[1] = 1.8
        self.size[2] = 0.8
    
    cpdef get_view_matrix(self):
        cdef double cos_p = cos(self.pitch)
        cdef double sin_p = sin(self.pitch)
        cdef double cos_y = cos(self.yaw)
        cdef double sin_y = sin(self.yaw)

        cdef float fx = <float>(cos_p * cos_y)
        cdef float fy = <float>sin_p
        cdef float fz = <float>(cos_p * sin_y)
        

        cdef float len_f = sqrt(fx*fx + fy*fy + fz*fz)
        cdef float zx = -fx / len_f
        cdef float zy = -fy / len_f
        cdef float zz = -fz / len_f
        

        cdef float ux = 0
        cdef float uy = 1
        cdef float uz = 0
        

        cdef float xx = uy*zz - uz*zy
        cdef float xy = uz*zx - ux*zz
        cdef float xz = ux*zy - uy*zx
        

        cdef float len_x = sqrt(xx*xx + xy*xy + xz*xz)
        xx = xx / len_x
        xy = xy / len_x
        xz = xz / len_x
        
        cdef float yx = zy*xz - zz*xy
        cdef float yy = zz*xx - zx*xz
        cdef float yz = zx*xy - zy*xx
        
        cdef float view[16]
        
        view[0] = xx
        view[1] = yx
        view[2] = zx
        view[3] = 0
        
        view[4] = xy
        view[5] = yy
        view[6] = zy
        view[7] = 0
        
        view[8] = xz
        view[9] = yz
        view[10] = zz
        view[11] = 0
        
        view[12] = -(xx*<float>self.x + xy*<float>self.y + xz*<float>self.z)
        view[13] = -(yx*<float>self.x + yy*<float>self.y + yz*<float>self.z)
        view[14] = -(zx*<float>self.x + zy*<float>self.y + zz*<float>self.z)
        view[15] = 1
        
        return view
        
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
        render.glEnable(render.GL_DEPTH_TEST)
        render.glEnable(render.GL_BLEND)
        render.glBlendFunc(render.GL_SRC_ALPHA, render.GL_ONE_MINUS_SRC_ALPHA)
        
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
        
        render.glUseProgram(self.shader)
        render.glUniformMatrix4fv(self.u_proj, 1, 0, ffi.cast("GLfloat *",<uintptr_t>&proj_view[0]))

    cpdef _create_white_texture(self):
        cdef cnp.uint8_t[:] data_view = np.array([255, 255, 255, 255], dtype=np.uint8)
        cdef unsigned int t_id
        render.glGenTextures(1,ffi.cast("GLuint *",<uintptr_t>&t_id))
        render.glBindTexture(render.GL_TEXTURE_2D, t_id)
        render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MIN_FILTER, render.GL_LINEAR)
        render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MAG_FILTER, render.GL_LINEAR)
        render.glTexImage2D(render.GL_TEXTURE_2D, 0, render.GL_RGBA, 1, 1, 0, render.GL_RGBA, render.GL_UNSIGNED_BYTE, ffi.cast("void *",<uintptr_t>&data_view[0]))
        return t_id

    cpdef camera(self): return self._cam

    cpdef _init_shaders(self):
        v_s = """
attribute vec3 pos; attribute vec3 col; attribute vec2 uv; attribute vec3 norm;
varying vec3 v_col; varying vec2 v_uv; varying vec3 v_norm; 
varying vec3 v_pos;

uniform mat4 proj, view, model;

void main() {
    v_col = col; 
    v_uv = uv; 
    
    vec4 worldPos = model * vec4(pos, 1.0);
    v_pos = worldPos.xyz; 
    
    v_norm = mat3(model) * norm;
    
    gl_Position = proj * view * worldPos;
}"""
        f_s = """
precision mediump float; 
varying vec3 v_col; varying vec2 v_uv; varying vec3 v_norm;
varying vec3 v_pos;

uniform sampler2D tex;
uniform vec3 u_lightPos;

void main() {
    vec3 lightDir = normalize(u_lightPos - v_pos); 
    float diff = max(dot(normalize(v_norm), lightDir), 0.25);
    
    gl_FragColor = texture2D(tex, v_uv) * vec4(v_col * diff, 1.0);
}"""
        self.shader = render. load_shaders(v_s,f_s)
        render.glUseProgram(self.shader)
        self.u_proj = render.glGetUniformLocation(self.shader, "proj")
        self.u_view = render.glGetUniformLocation(self.shader, "view")
        self.u_model = render.glGetUniformLocation(self.shader, "model")
        self.u_lightpos = render.glGetUniformLocation(self.shader, "u_lightPos")
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

        self.gui_shader = render.load_shaders(v_code, f_code)

    cpdef load_texture(self, path):
        cdef unsigned int t_id
        cdef const unsigned char[::1] pixel_view
        if not os.path.exists(path): return self.default_tex
        try:
            img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
            render.glGenTextures(1,ffi.cast("GLuint *",<uintptr_t>&t_id))
            render.glBindTexture(render.GL_TEXTURE_2D, t_id)
            # фильтрация для MAG_FILTER
            render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MIN_FILTER, render.GL_LINEAR)
            render.glTexParameteri(render.GL_TEXTURE_2D, render.GL_TEXTURE_MAG_FILTER, render.GL_LINEAR)
            pixel_view = img.tobytes()
            render.glTexImage2D(render.GL_TEXTURE_2D, 0, render.GL_RGBA, img.width, img.height, 0, render.GL_RGBA, render.GL_UNSIGNED_BYTE, ffi.cast("void *",<uintptr_t>&pixel_view[0]))
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
        render.glGenBuffers(1, ffi.cast("GLuint *",<uintptr_t>&vbo))
        render.glBindBuffer(render.GL_ARRAY_BUFFER, vbo)
        cdef size_t buffer_size = final_v.size() * sizeof(float)
        render.glBufferData(
            render.GL_ARRAY_BUFFER, 
            buffer_size, 
            ffi.cast("void *",<uintptr_t>&final_v[0]), 
            render.GL_STATIC_DRAW
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
        render.glClear(render.GL_COLOR_BUFFER_BIT | render.GL_DEPTH_BUFFER_BIT)
    cpdef ScreenColor(self, float r, float g, float b, float a=1.0):
        render.glClearColor(r, g, b, a)

    cpdef draw(self, obj):
        if not obj: return
        render.begin_draw(self.shader)
        render.set_matrices(self._cam,obj,self.u_model,self.u_view)
        render.set_sun(self.u_lightpos, self.sun)
        render.setup_vertex(self.shader,obj.vbo)
        render.draw_mesh(obj)

    cpdef draw_gui(self, unsigned int texture_id, float x, float y, float w, float h):
        cdef float sw = <float>self.width
        cdef float sh = <float>self.height
        cdef float[:] view = np.array([
            0.0, 0.0,  0.0, 0.0, 
            1.0, 0.0,  1.0, 0.0, 
            0.0, 1.0,  0.0, 1.0, 
            1.0, 1.0,  1.0, 1.0  
        ], dtype=np.float32)

        render.glUseProgram(self.gui_shader)
        render.glDisable(render.GL_DEPTH_TEST)
        render.glBindBuffer(render.GL_ARRAY_BUFFER, 0)
        cdef int rect_loc = render.glGetUniformLocation(self.gui_shader, "u_rect")
        render.glUniform4f(rect_loc, x, y, w, h)
        cdef float proj[16]
        for i in range(16): proj[i] = 0.0
        proj[0] = 2.0 / sw;    proj[12] = -1.0
        proj[5] = -2.0 / sh;   proj[13] = 1.0
        proj[10] = 1.0;        proj[15] = 1.0

        cdef int proj_loc = render.glGetUniformLocation(self.gui_shader, "u_proj")
        render.glUniformMatrix4fv(proj_loc, 1, 0, proj)
        render.glActiveTexture(0x84C0)
        render.glBindTexture(render.GL_TEXTURE_2D, texture_id)
        render.glUniform1i(render.glGetUniformLocation(self.gui_shader, "u_texture"), 0)

        cdef int pos_loc = render.glGetAttribLocation(self.gui_shader, "position")
        cdef int uv_loc = render.glGetAttribLocation(self.gui_shader, "texCoord")
        
        render.glEnableVertexAttribArray(pos_loc)
        render.glVertexAttribPointer(pos_loc, 2, 0x1406, 0, 16, ffi.cast("GLfloat *",<uintptr_t>&view[0]))
        render.glEnableVertexAttribArray(uv_loc)
        render.glVertexAttribPointer(uv_loc, 2, 0x1406, 0, 16, ffi.cast("GLfloat *",<uintptr_t>&view[2]))

        render.glDrawArrays(0x0005, 0, 4)
        render.glEnable(render.GL_DEPTH_TEST)

