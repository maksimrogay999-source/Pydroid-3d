from cffi import FFI
ffi = FFI()
vk = ffi.dlopen("libvulkan.so")
ffi.cdef("int vkEnumerateInstanceVersion(unsigned int* pApiVersion);")
version = ffi.new("unsigned int *")
result = vk.vkEnumerateInstanceVersion(version)
if result == 0:
    v = version[0]
    major = v >> 22
    minor = (v >> 12) & 0x3FF
    patch = v & 0xFFF
    print(f"Vulkan версия: {major}.{minor}.{patch}")
else:
    print("Vulkan не найден")