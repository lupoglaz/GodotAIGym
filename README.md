# GodotAIGym
![logo](docs/Fig/GodotGymLogo.png)


Make your Godot project into OpenAI Gym environment to train RL models with PyTorch. This project only runs on Linux systems for now.

## Requirements
1. Godot Engine version == 3.2 compiled from source (not tested with the later versions)
2. Boost interprocess and time (apt install libboost-container-dev libboost-system-dev libboost-locale-dev)
3. Pytorch version == 1.10
4. Python setuptools

## Installation
First, in **setup.py** change the variable **GODOT_PATH** to the root directory of godot engine source. Then run:
```bash
python setup.py
```
This script does several things:
1. Downloads libtorch (v1.10) cpu only version, unpacks it
2. Copies **GodotSharedMemory** module and compiles standard godot editor (x11 platform).
3. Compiles x11 export template and dev tools
4. Installs python module **GodotEnv** that is used to communicate with the engine.

## Docs
[InvPendulum](https://lupoglaz.github.io/GodotAIGym/tutorial_basic.html)
tutorial shows how to make an environment, speed up its execution, train a model and deploy back to the engine.

[API](https://lupoglaz.github.io/GodotAIGym/API.html) lists classes and function in python and godot.

# TODO
1. Check if it's possible to use GDNative + shared libs instead of the engine recompilation.
2. Windows compatibility