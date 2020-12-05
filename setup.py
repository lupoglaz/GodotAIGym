import os
import sys
import argparse
from shutil import copyfile, copytree, rmtree, which
from urllib import request
from zipfile import ZipFile
from pathlib import Path

GODOT_PATH = os.environ["GODOT_PATH"]

def download_unpack(rewrite=False):
	# url = 'https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip'
	url = 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcpu.zip'
	if (not Path('libtorch.zip').exists()) or rewrite:
		print('Downloading libtorch')
		filedata = request.urlopen(url)
		datatowrite = filedata.read()
		with open('libtorch.zip', 'wb') as f:
			f.write(datatowrite)

	libtorch_path = 'GodotModule/libtorch'
	if Path(libtorch_path).exists():
		rmtree(libtorch_path)

	print('Extracting libtorch')
	with ZipFile('libtorch.zip', 'r') as zipObj:
		zipObj.extractall(path='GodotModule')
	

def compile_godot(godot_root, platform='x11', tools='yes', target='release_debug', bits=64):
	current_path = os.getcwd()
	os.chdir(godot_root)
	os.system(f"scons platform={platform} tools={tools} target={target} bits={bits}")
	os.chdir(current_path)

def install_module(godot_root, rewrite=False):
	module_dir = os.path.join(godot_root, 'modules/GodotSharedMemory')
	if (not os.path.exists(module_dir)):
		copytree('GodotModule', module_dir)
	elif rewrite:
		rmtree(module_dir)
		copytree('GodotModule', module_dir)
	
def install_python_module():
	current_path = os.getcwd()
	os.chdir('PythonModule')
	python_command = 'python'
	if which(python_command) is None:
		python_command += '3'
	os.system(python_command + ' setup.py install --user')
	os.chdir(current_path)

if __name__=='__main__':
	download_unpack(rewrite=False)
	install_module(godot_root=GODOT_PATH, rewrite=True)
	install_python_module()
	compile_godot(godot_root=GODOT_PATH, platform='x11', tools='yes', target='release_debug', bits=64)
	compile_godot(godot_root=GODOT_PATH, platform='x11', tools='no', target='release_debug', bits=64)
	compile_godot(godot_root=GODOT_PATH, platform='server', tools='no', target='release_debug', bits=64)
