cdef extern from "SDL2/SDL.h" nogil:
    struct SDL_Window:
        pass
    
    ctypedef void* SDL_GLContext

    unsigned int SDL_INIT_VIDEO
    unsigned int SDL_INIT_EVERYTHING
    unsigned int SDL_WINDOW_OPENGL

    
    int SDL_Init(unsigned int flags)
    SDL_Window* SDL_CreateWindow(
        const char* title, 
        int x, 
        int y, 
        int w, 
        int h, 
        unsigned int flags
    )
    SDL_GLContext SDL_GL_CreateContext(SDL_Window* window)
    void SDL_Delay(unsigned int ms)
    void SDL_GL_SwapWindow(SDL_Window* window)
    void SDL_ShowWindow(SDL_Window* window)
