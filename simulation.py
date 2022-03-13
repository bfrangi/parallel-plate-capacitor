from initial_parameters import *
import numpy as np

# SPACE AND TIME VECTORS
def generate_mesh(vec_min, vec_max, Mv, Dv):
	v = []
	#Dv = (vec_max - vec_min) / (Mv - 1)
	for k in range(Mv):
		v.append( vec_min + k*Dv )
	return np.array(v)

def create_mesh(x, y, z):
	mesh = []
	for j in range(len(x)):
		this_x = []
		for k in range(len(y)):
			this_y = []
			for l in range(len(z)):
				this_y.append(initial_potential(j, k, l))
			this_x.append(this_y)
		mesh.append(this_x)
	return mesh

def initial_potential(j, k, l):
	edge_x = [0, Mx - 1]
	edge_y = [0, My - 1]
	edge_z = [0, Mz - 1]

	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = range(int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = range(int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)

	if j in edge_x or k in edge_y or l in edge_z:
		return Vbox
	elif k in plates_y and l in plates_z:
		if j == plate1_x:
			return V1
		elif j == plate2_x:
			return V2
		else: 
			return 0
	else:
		return 0

def find_center(mesh):
	x_len = len(mesh)
	y_len = len(mesh[0])
	z_len = len(mesh[0][0])
	return (int(x_len/2), int(y_len/2), int(z_len/2))
	
def print_z_cut(l, mesh, show_zeros=False):
	x_len = len(mesh)
	y_len = len(mesh[0])
	for j in range(x_len):
		for k in range(y_len):
			if show_zeros:
				print(mesh[j][k][l], end=" ")
			else:
				m = mesh[j][k][l]
				if m != 0:
					print(m, end=" ")
				else:
					print(" ", end="")
		print("")

def print_y_cut(k, mesh, show_zeros=False):
	x_len = len(mesh)
	z_len = len(mesh[0][0])
	for j in range(x_len):
		for l in range(z_len):
			if show_zeros:
				print(mesh[j][k][l], end=" ")
			else:
				m = mesh[j][k][l]
				if m != 0:
					print(m, end=" ")
				else:
					print(" ", end="")
		print("")

def print_x_cut(j, mesh, show_zeros=False):
	y_len = len(mesh[0])
	z_len = len(mesh[0][0])
	for k in range(y_len):
		for l in range(z_len):
			if show_zeros:
				print(mesh[j][k][l], end=" ")
			else:
				m = mesh[j][k][l]
				if m!=0:
					print(m, end=" ")
				else:
					print(" ", end="")
		print("")

def stencil_average(j, k, l, mesh):
	top = mesh[j][k][l + 1]
	bottom = mesh[j][k][l - 1]
	left = mesh[j - 1][k][l]
	right = mesh[j + 1][k][l]
	forward = mesh[j][k + 1][l]
	backward = mesh[j][k - 1][l]
	current = mesh[j][k][l]

	new = (top + bottom + left + right + forward + backward)/6
	residual = abs(current - new)

	return (top + bottom + left + right + forward + backward)/6, residual

def updated_potential(j, k, l, mesh):
	edge_x = [0, Mx - 1]
	edge_y = [0, My - 1]
	edge_z = [0, Mz - 1]

	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = range(int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = range(int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)

	if j in edge_x or k in edge_y or l in edge_z:
		return Vbox, 0
	elif k in plates_y and l in plates_z:
		if j == plate1_x:
			return V1, 0# (Zero is the residual)
		elif j == plate2_x:
			return V2, 0# (Zero is the residual)
		else:
			return stencil_average(j, k, l, mesh)
	else:
		return stencil_average(j, k, l, mesh)

def update_potential(mesh):
	x_len = len(mesh)
	y_len = len(mesh[0])
	z_len = len(mesh[0][0])
	max_res = 0
	for j in range(x_len):
		for k in range(y_len):
			for l in range(z_len):
				mesh[j][k][l], residual = updated_potential(j, k, l, mesh)
				if residual > max_res:
					max_res = residual

	return mesh, max_res

# MAIN FUNCTION
if __name__=="__main__":
	x = generate_mesh(x_min, x_max, Mx, Dx)
	y = generate_mesh(y_min, y_max, My, Dy)
	z = generate_mesh(z_min, z_max, Mz, Dz)
	#mesh = lambda xi, yi, zi: (x[xi], y[yi], z[zi])

	mesh = create_mesh(x, y, z)
	mesh_center = find_center(mesh)	

	prev_residual = 0
	residual = 1000# (any value different to the prev_residual)
	reached_tolerance = False
	iteration_number = 1
	while residual != prev_residual and not reached_tolerance:
		prev_residual = residual
		if iteration_number == 1:
			print("1. Computing relaxation of potential...")
		else:
			print(str(iteration_number) + ". Computing relaxation of potential... (Previous Residual: " + str(round(prev_residual, 3)) + ")")
		mesh, residual = update_potential(mesh)
		iteration_number += 1
		if residual < Rtol:
			reached_tolerance = True	

	print_z_cut(mesh_center[2], mesh)



	#print_z_cut(mesh_center[2], mesh)
	#print_y_cut(mesh_center[1], mesh)
	#print_x_cut(int((x1-x_min)/Dx), mesh)
	#print_x_cut(int((x2-x_min)/Dx), mesh)
	#print_x_cut(mesh_center[0], mesh)

