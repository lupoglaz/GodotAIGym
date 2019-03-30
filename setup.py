import os 
import sys 
from shutil import copyfile

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
	
def install_templates(username, version, template_name, godot_bin_dir):
	godot_share_dir = "/home/%s/.local/share/godot"%username
	if not os.path.exists(godot_share_dir):
		os.mkdir(godot_share_dir)
	godot_templates_dir = os.path.join(godot_share_dir, "templates")
	if not os.path.exists(godot_templates_dir):
		os.mkdir(godot_templates_dir)
	
	godot_ver_templates_dir = os.path.join(godot_share_dir, version)
	if not os.path.exists(godot_ver_templates_dir):
		os.mkdir(godot_ver_templates_dir)

	template_file_src = os.path.join(godot_bin_dir, template_name)
	template_file_dst = os.path.join(godot_ver_templates_dir, template_name)
	if os.path.exists(template_file_src):
		copyfile(template_file_src, template_file_dst)
		print('Copy: %s --> %s'%(template_file_src, template_file_dst))
	else:
		print('File %s not found. Compile godot first'%(template_file_src))

if __name__=='__main__':
	
	patch_script(	filename = '../../drivers/gl_context/SCsub',
					patched_line = 'if (env["platform"] in ["haiku", "osx", "windows", "x11", "x11_shared"]):\n'
				)

	install_templates(	username="lupoglaz", 
						version="3.1",
						template_name="godot_shared.x11_shared.opt.64",
						godot_bin_dir="/home/lupoglaz/Projects/godot/bin")


	


	

