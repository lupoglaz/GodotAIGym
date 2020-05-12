import os
import sys
import argparse
from shutil import copyfile, copytree, rmtree

GODOT_PATH = os.environ["GODOT_PATH"]

def patch_script(godot_root, filename):
	current_path = os.getcwd()
	patch_path = os.path.join(current_path, filename)
	os.chdir(godot_root)
	os.system("git am %s"%patch_path)
	os.chdir(current_path)

def compile_godot(godot_root, platform='x11', target='release_debug', bits=64):
	current_path = os.getcwd()
	os.chdir(godot_root)
	os.system("scons platform=%s tools=yes target=%s bits=%d"%(platform, target, bits))
	os.system("scons platform=%s tools=no target=%s bits=%d"%(platform, target, bits))
	os.chdir(current_path)

def install_module(godot_root, compile=True, rewrite=False):
	module_dir = os.path.join(godot_root, 'modules/GodotSharedMemory')
	if (not os.path.exists(module_dir)):
		copytree('GodotModule', module_dir)
	elif rewrite:
		rmtree(module_dir)
		copytree('GodotModule', module_dir)
	
	if compile:
		compile_godot(godot_root, platform='x11', target='release_debug', bits=64)

def install_python_module():
	current_path = os.getcwd()
	os.chdir('PythonModule')
	os.system('python setup.py install')
	os.chdir(current_path)

if __name__=='__main__':
	# patch_script(godot_root=GODOT_PATH, filename="PhysicsPatch/patch")
	install_module(godot_root=GODOT_PATH, compile=True, rewrite=True)
	install_python_module()
