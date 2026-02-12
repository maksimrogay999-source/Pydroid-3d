cdef extern from "<GLES2/gl2.h>":
    unsigned int GL_DEPTH_TEST
    unsigned int GL_TEXTURE_2D
    unsigned int GL_ARRAY_BUFFER
    unsigned int GL_FALSE
    unsigned int GL_TEXTURE_MIN_FILTER
    unsigned int GL_TEXTURE_MAG_FILTER
    unsigned int GL_LINEAR
    unsigned int GL_RGBA
    unsigned int GL_UNSIGNED_BYTE
    unsigned int GL_ARRAY_BUFFER
    unsigned int GL_STATIC_DRAW
    unsigned int GL_TRIANGLES
    unsigned int GL_COLOR_BUFFER_BIT
    unsigned int GL_DEPTH_BUFFER_BIT
    unsigned int GL_VERTEX_SHADER
    unsigned int GL_FRAGMENT_SHADER
    unsigned int GL_FLOAT


    void glEnable(unsigned int cap)
    void glDisable(unsigned int cap)
    void glUseProgram(unsigned int program)
    void glUniformMatrix4fv(int location, int count, unsigned char transpose, const float* value)
    void glBindTexture(unsigned int target, unsigned int texture)
    void glGenTextures(int n, unsigned int *textures)
    void glBindBuffer(unsigned int target, unsigned int buffer)
    void glGenBuffers(int n, unsigned int* buffers)
    void glTexParameteri(unsigned int target, unsigned int pname, int param)
    void glTexImage2D(unsigned int target, int level, int internalformat,int width, int height, int border,unsigned int format, unsigned int type, const void* pixels)
    int glGetUniformLocation(unsigned int program, const char* name)
    void glBufferData(unsigned int target, long int size, const void* data, unsigned int usage)
    int glGetAttribLocation(unsigned int program, const char* name)
    void glEnableVertexAttribArray(unsigned int index)
    void glVertexAttribPointer(unsigned int index, int size, unsigned int type, unsigned char normalized, int stride, const void* pointer)
    void glDrawArrays(unsigned int mode, int first, int count)
    void glClear(unsigned int mask)
    void glClearColor(float red, float green, float blue, float alpha)
    unsigned int glCreateShader(unsigned int type)
    void glShaderSource(unsigned int shader, int count, const char** string, int* length)
    void glCompileShader(unsigned int shader)
    unsigned int glCreateProgram()
    void glAttachShader(unsigned int program, unsigned int shader)
    void glLinkProgram(unsigned int program)