from initial_parameters import *
import numpy as np

# SPACE AND TIME VECTORS
def generate_mesh(vec_min, vec_max, Mv):
	v = []
	Dv = (vec_max - vec_min) / (Mv - 1)
	for k in range(Mv):
		v.append( vec_min + k*Dv )
	return np.array(v)

def create_mesh(x, y, z):
	#x = [num for num in range(-5, 6)]
	#y = [num for num in range(-5, 6)]
	#z = [num for num in range(-5, 6)]
	mesh = []

	for i in range(len(x)):
		this_x = []
		for j in range(len(y)):
			this_y = []
			for k in range(len(z)):
				this_y.append(potential(i, j, k))
				count += 1
			this_x.append(this_y)
		mesh.append(this_x)
	return mesh

def potential(i, j, k):
	return 0

# MAIN FUNCTION
if __name__=="__main__":
	x = generate_mesh(x_min, x_max, Mx)
	y = generate_mesh(y_min, y_max, My)
	z = generate_mesh(z_min, z_max, Mz)
	#mesh = lambda xi, yi, zi: (x[xi], y[yi], z[zi])
	
	mesh = create_mesh(x, y, z)

	#for i in range(len(x)):
	#	for j in range(len(y)):
	#		for k in range(len(z)):
	#			print(mesh(i, j, k))
