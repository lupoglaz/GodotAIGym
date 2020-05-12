from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import sysconfig

if __name__=='__main__':
	
	Packages = ['GodotEnv']

	GodotEnv = CppExtension('_GodotEnv', 
					sources = [ 'src/cGodotSharedInterface.cpp', 
                                'src/main.cpp'
                            ],
					include_dirs = ['src'],
					libraries = ['boost_locale', 'boost_system', 'stdc++', 'pthread', 'rt'],
					extra_compile_args=[])
	
	setup(	name='GodotEnv',
			version="0.1",
			ext_modules=[	GodotEnv
						],
			cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
			
			packages = Packages,
			author="Georgy Derevyanko",
			author_email="georgy.derevyanko@gmail.com",
			description="Shared memory connection to godot",
			license="MIT",
			keywords="godot, reinforcement learning, environment",
			url="https://github.com/lupoglaz/GodotSharedMemory",
			project_urls={
				"Bug Tracker": "https://github.com/lupoglaz/GodotSharedMemory/issues",
				"Documentation": "https://github.com/lupoglaz/GodotSharedMemory/tree/Release/Doc",
				"Source Code": "https://github.com/lupoglaz/GodotSharedMemory",
			})
