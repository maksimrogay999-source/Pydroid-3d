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