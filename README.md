# GodotAIGym
![logo](docs/Fig/GodotGymLogo.png)


Make your Godot project into OpenAI Gym environment to train RL models with PyTorch. This project only runs on Linux systems.

## Requirements
1. Godot Engine version >= 3.2 compiled from source
2. Boost interprocess (apt install libboost-container-dev libboost-system-dev libboost-locale-dev)
3. Pytorch version >= 1.5
4. Python setuptools

## Installation
First, in **setup.py** change the variable **GODOT_PATH** to the root directory of godot engine source. Then run:
```bash
python setup.py
```
This script does several things:
1. Downloads libtorch cpu only version, unpacks it
2. Copies **GodotSharedMemory** module and compiles standard godot editor (x11 platform).
3. Compiles x11 export template, dev tools and server templates
4. Installs python module **GodotEnv** that is used to communicate with the engine.

## Examples
There are several example environments in the directory **Tutorials**:

**InvPendulum** shows the example of environment that relies heavily on the physics engine

**LunarLander** more complex example that shows how to randomize environment upon reset

## Tutorial and API
[Tutorial](https://lupoglaz.github.io/GodotGymAI/tutorial.html)

[API](https://lupoglaz.github.io/GodotGymAI/API.html)

# TODO
1. **Models deployment**: add module to load your traced model into godot engine
2. **Learning from pixels**: passing godot viewport rendering as a torch tensor
3. **Complete tutorial**: tutorial about complete workflow, from empty godot project to training to deployment
