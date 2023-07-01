import os
import sys
import argparse
from shutil import copyfile, copytree, rmtree
from urllib import request
from zipfile import ZipFile
from pathlib import Path

# GODOT_PATH = os.environ["GODOT_PATH"]
# GODOT_PATH = "/home/lupoglaz/Projects/godot32"

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

if __name__=='__main__':
	# assert os.path.exists(GODOT_PATH)
	
	download_unpack(rewrite=True)
	install_python_module()
	if Path("GDExtension").exists():
		os.system("rm -r GDExtension")
	os.system('cmake -S . -B ./GDExtension')
	os.system('cmake --build ./GDExtension')

