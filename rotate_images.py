import math
import os
import os.path
import sys
from random import randint
from PIL import Image

def image_rotate_random(in_file_path, out_dir):

	print(in_file_path)
	in_file = os.path.basename(in_file_path)
	file_name = ''.join(in_file.split('.')[:-1])

	img = Image.open(in_file_path)
	sx, sy = img.size
	cx, cy = sx/2.0, sy/2.0
	d = min(cx, cy)
	a = d*math.sqrt(2)
	area = (cx - a/2, cy - a/2, cx + a/2, cy + a/2)

	print('sx={0}, sy={1}'.format(sx,sy))
	print('cx={0}, cy={1}'.format(cx,cy))
	print('d={0:.3f}, a={1:.3f}'.format(d,a))

	for i in range(0, 12):

		angle = i*30 + randint(0,29)
		print(angle)

		img_rot = img.rotate(angle)
		box = img_rot.crop(area)

		#box.show()
		out_file = out_dir + '/' \
			+ '{0}_{1:03d}.jpg'.format(file_name, angle)
		box.save(out_file)


def rotate_all_in_dir_random(in_dir, out_dir):

	files = os.listdir(in_dir)
	
	for file_name in files:
		file_path = in_dir + '/' + file_name
		image_rotate_random(file_path, out_dir)


def rotate_images_with_angles(in_dir, out_dir, file_names, angles):

	for i, file_name in enumerate(file_names):

		in_file_path = in_dir + '/' + file_name
		out_file_path = out_dir + '/' + file_name
		img = Image.open(in_file_path)
		angle = -angles[i]
		img_rot = img.rotate(angle)
		img_rot.save(out_file_path)
		print('{0}: {1} - angle={2}'.format(i, file_name, angle))

#-----------------------------------
if __name__ == '__main__':
	
	in_dir = 'in'
	out_dir = 'out'

	in_dir = in_dir.rstrip('/')
	out_dir = out_dir.rstrip('/')
	os.system('mkdir -p {0}'.format(out_dir))

	rotate_all_in_dir_random(in_dir, out_dir)