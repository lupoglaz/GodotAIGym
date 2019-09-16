import os
import sys
import argparse
from shutil import copyfile, copytree, rmtree

GODOT_PATH = "/home/lupoglaz/Projects/godot"

def patch_script(filename, patched_line):
	lines = []
	modified = False
	with open(filename, 'r') as fin:
		for line in fin:
			lines.append(line)
			if line.find('platform')!=-1 and line != patched_line:
				lines[-1] = patched_line
				modified = True
	
	if modified:
		print("Patched %s, line = %s"%(filename, patched_line))
		with open(filename, 'w') as fout:
			for line in lines:
				fout.write(line)
	else:
		print("No modifications needed")

def compile_godot(godot_root, platform='x11', target='release_debug', bits=64, tools=False):
	current_path = os.getcwd()
	os.chdir(godot_root)
	if tools:
		os.system("scons platform=%s tools=yes target=%s bits=%d"%(platform, target, bits))
	else:    
		os.system("scons platform=%s tools=no target=%s bits=%d"%(platform, target, bits))
	os.chdir(current_path)

def install_platform(godot_root, compile=True, rewrite=False):
	platform_dir = os.path.join(godot_root, 'platform/x11_shared')
	if (not os.path.exists(platform_dir)):
		copytree('GodotPlatform', platform_dir)
	elif rewrite:
		rmtree(platform_dir)
		copytree('GodotPlatform', platform_dir)
		
	patch_script(	filename = os.path.join(godot_root, 'drivers/gl_context/SCsub'),
					patched_line = 'if (env["platform"] in ["haiku", "osx", "windows", "x11", "x11_shared"]):\n'
				)
	if compile:
		compile_godot(godot_root, platform='x11_shared', target='release_debug', bits=64, tools=False)

def install_module(godot_root, compile=True, rewrite=False):
	module_dir = os.path.join(godot_root, 'modules/GodotSharedMemory')
	if (not os.path.exists(module_dir)):
		copytree('GodotModule', module_dir)
	elif rewrite:
		rmtree(module_dir)
		copytree('GodotModule', module_dir)
	
	if compile:
		compile_godot(godot_root, platform='x11', target='release_debug', bits=64, tools=True)

def install_python_module():
	current_path = os.getcwd()
	os.chdir('PythonModule')
	os.system('python setup.py install')
	os.chdir(current_path)

if __name__=='__main__':
	install_module(godot_root=GODOT_PATH)
	install_platform(godot_root=GODOT_PATH)
	install_python_module()