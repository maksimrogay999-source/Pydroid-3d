cdef extern from "<GLES2/gl2.h>":
    unsigned int GL_DEPTH_TEST
    unsigned int GL_TEXTURE_2D
    void glEnable(unsigned int cap)
    void glDisable(unsigned int cap)
    void glUseProgram(unsigned int program)
    void glUniformMatrix4fv(int location, int count, unsigned char transpose, const float* value)
    void glBindTexture(unsigned int target, unsigned int texture)

