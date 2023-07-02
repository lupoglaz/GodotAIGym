import os
import sys
import argparse
from shutil import copyfile, copytree, rmtree
from urllib import request
from zipfile import ZipFile
from pathlib import Path

# GODOT_PATH = os.environ["GODOT_PATH"]
# GODOT_PATH = "/home/lupoglaz/Projects/godot32"

def generate_config(entry_symbol, library_filename:str):
	return f'[configuration]\n\n\
entry_symbol = "{entry_symbol}"\n\
compatibility_minimum = 4.0\
\n\n[libraries]\n\n\
linux.debug.x86_64 = "res://bin/{library_filename}"\n\
linux.release.x86_64 = "res://bin/{library_filename}"'

def download_unpack(rewrite=False):
	url = 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.1%2Bcpu.zip'
	# url = 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip'
	if (not Path('libtorch.zip').exists()) or rewrite:
		print('Downloading libtorch')
		filedata = request.urlopen(url)
		datatowrite = filedata.read()
		with open('libtorch.zip', 'wb') as f:
			f.write(datatowrite)

	if (not Path('libtorch').exists()) or rewrite:
		print('Extracting libtorch')
		with ZipFile('libtorch.zip', 'r') as zipObj:
   			zipObj.extractall(path='.')
	
def install_python_module():
	current_path = os.getcwd()
	os.chdir('PythonModule')
	os.system('python setup.py install')
	os.chdir(current_path)

def gen_output(output_dir=Path("bin"), lib_name="SharedMemory", entry_symbol="sharedmemory_library_init", tool='cmake'):
	if not(output_dir.exists()):
		output_dir.mkdir()
	if tool=='cmake':
		lib_file = f'lib{lib_name}.so'
	elif tool=='scons':
		lib_file = f'libgd{lib_name.lower()}.linux.template_debug.x86_64.so'
		
	os.system(f"cp build/{lib_file} {output_dir.as_posix()}/{lib_file}")
	
	with open(f"{output_dir.as_posix()}/{lib_name}.gdextension", "wt") as fout:
		fout.write(generate_config(entry_symbol, lib_file))

def build_cmake():
	if Path("bild").exists():
		os.system("rm -r build")
	os.system('cmake -S . -B ./build')
	os.system('cmake --build ./build')

	output_dir = Path('bin')
	gen_output(output_dir, "SharedMemory", "sharedmemory_library_init", tool='cmake')
	gen_output(output_dir, "TorchModel", "torchmodel_library_init", tool='cmake')

def build_scons():
	if Path("bild").exists():
		os.system("rm -r build")
	os.system('scons platform=linux')
	
	output_dir = Path('bin')
	gen_output(output_dir, "SharedMemory", "sharedmemory_library_init", tool='scons')
	gen_output(output_dir, "TorchModel", "torchmodel_library_init", tool='scons')
	


if __name__=='__main__':
	# assert os.path.exists(GODOT_PATH)
	
	# download_unpack(rewrite=True)
	# install_python_module()
	build_scons()
	

