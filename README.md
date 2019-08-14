# GodotSharedMemory
Communicating with godot engine through shared memory

# Installation
## Clone module:
```bash
git clone --single-branch --branch module https://github.com/lupoglaz/GodotSharedMemory.git GodotSharedMemory
```
## Clone platform:
```bash
git clone --single-branch --branch x11_shared https://github.com/lupoglaz/GodotSharedMemory.git x11_shared
```

## Recompile godot:
```bash
scons platform=x11_shared tools=no target=release bits=64
```
