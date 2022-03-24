from initial_parameters import *
import numpy as np
import re
from numba import jit, njit
from time import time as t
import matplotlib.pyplot as plt
import os

# SPACE AND TIME VECTORS
#@njit #---> Works, but as it is only called once, it is faster to
def generate_vector(vec_min:float, vec_max:float, Mv:int, Dv:float, v:np.array=np.array([])) -> np.array:
	#v = np.array([])
	#Dv = (vec_max - vec_min) / (Mv - 1)
	for k in range(Mv):
		v = np.append(v, [vec_min + k*Dv])
	return v

from numba.typed import List

#@njit #---> Not working yet
def create_mesh(x:np.array, y:np.array, z:np.array, mesh:list=[], this_x:list=[], this_y:list=[]) -> np.array:
	mesh = []
	for j in range(len(x)):
		this_x = []
		for k in range(len(y)):
			this_y = []
			for l in range(len(z)):
				this_y.append(initial_potential(j, k, l))
				#this_y.append(10.)
			this_x.append(this_y)
		mesh.append(this_x)
	return np.array(mesh)

#@njit #---> Works, but is no faster than with no @njit
def initial_potential(j:int, k:int, l:int) -> float:
	edge_x = [0, Mx - 1]
	edge_y = [0, My - 1]
	edge_z = [0, Mz - 1]

	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = range(int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = range(int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)

	if j in edge_x or k in edge_y or l in edge_z:
		return float(Vbox)
	elif k in plates_y and l in plates_z:
		if j == plate1_x:
			return float(V1)
		elif j == plate2_x:
			return float(V2)
		else: 
			return 0.
	else:
		return 0.

def find_center(mesh):
	x_len = len(mesh)
	y_len = len(mesh[0, :, :])
	z_len = len(mesh[0, 0, :])
	return (int(x_len/2), int(y_len/2), int(z_len/2))
	
def print_z_cut(l, mesh, show_zeros=False, rounding='no'):
	x_len = len(mesh)
	y_len = len(mesh[0, :, :])
	for j in range(x_len):
		for k in range(y_len):
			if show_zeros:
				if rounding=='no':
					print(mesh[j, k, l], end=" ")
				else:
					print(round(mesh[j, k, l], rounding), end=" ")
			else:
				m = mesh[j, k, l]
				if m != 0:
					if rounding=='no':
						print(m, end=" ")
					else:
						print(round(m, rounding), end=" ")
				else:
					print(" ", end="")
		print("")

def print_y_cut(k, mesh, show_zeros=False, rounding='no'):
	x_len = len(mesh)
	z_len = len(mesh[0, 0, :])
	for j in range(x_len):
		for l in range(z_len):
			if show_zeros:
				if rounding=='no':
					print(mesh[j, k, l], end=" ")
				else:
					print(round(mesh[j, k, l], rounding), end=" ")
			else:
				m = mesh[j, k, l]
				if m != 0:
					if rounding=='no':
						print(m, end=" ")
					else:
						print(round(m, rounding), end=" ")
				else:
					print(" ", end="")
		print("")

def print_x_cut(j, mesh, show_zeros=False, rounding='no'):
	y_len = len(mesh[0])
	z_len = len(mesh[0, 0, :])
	for k in range(y_len):
		for l in range(z_len):
			if show_zeros:
				if rounding=='no':
					print(mesh[j, k, l], end=" ")
				else:
					print(round(mesh[j, k, l], rounding), end=" ")
			else:
				m = mesh[j, k, l]
				if m != 0:
					if rounding=='no':
						print(m, end=" ")
					else:
						print(round(m, rounding), end=" ")
				else:
					print(" ", end="")
		print("")

@njit # Works, faster with njit
def stencil_average(j:int, k:int, l:int, mesh:np.array) -> tuple:
	top = mesh[j, k, l + 1]
	bottom = mesh[j, k, l - 1]
	left = mesh[j - 1, k, l]
	right = mesh[j + 1, k, l]
	forward = mesh[j, k + 1, l]
	backward = mesh[j, k - 1, l]
	current = mesh[j, k, l]

	new = (top + bottom + left + right + forward + backward)/6
	residual = abs(current - new)

	return (top + bottom + left + right + forward + backward)/6, residual

@njit # Works, much faster with njit
def updated_potential(j:int, k:int, l:int, mesh:np.array) -> tuple:
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
			return float(V1), 0.# (Zero is the residual)
		elif j == plate2_x:
			return float(V2), 0.# (Zero is the residual)
		else:
			return stencil_average(j, k, l, mesh)
	else:
		return stencil_average(j, k, l, mesh)

def update_potential(mesh:np.array) -> tuple:
	edge_x = [0, Mx - 1]
	edge_y = [0, My - 1]
	edge_z = [0, Mz - 1]

	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = (int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = (int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)

	x_len = len(mesh)
	y_len = len(mesh[0, :, :])
	z_len = len(mesh[0, 0, :])
	new_mesh = np.empty((x_len, y_len, z_len))
	new_mesh[1:x_len-1, 1: y_len-1,1:z_len-1] = (mesh[0:x_len-2, 1: y_len-1,1:z_len-1] + \
												mesh[2:x_len, 1: y_len-1,1:z_len-1] + \
												mesh[1:x_len-1, 0: y_len-2,1:z_len-1] + \
												mesh[1:x_len-1, 2: y_len,1:z_len-1] + \
												mesh[1:x_len-1, 1: y_len-1,0:z_len-2] + \
												mesh[1:x_len-1, 1: y_len-1,2:z_len])/6

	new_mesh[plate1_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V1)
	new_mesh[plate2_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V2)
	max_res = np.amax(abs(mesh - new_mesh))

	return new_mesh, max_res

@njit # Works, much faster with njit
def update_potential_old(mesh:np.array) -> tuple:
	x_len = len(mesh)
	y_len = len(mesh[0, :, :])
	z_len = len(mesh[0, 0, :])
	max_res = 0
	for j in range(x_len):
		for k in range(y_len):
			for l in range(z_len):
				mesh[j, k, l], residual = updated_potential(j, k, l, mesh)
				if residual > max_res:
					max_res = residual
	return mesh, max_res

def export_matrix(matrix, filename='potential_matrix.npy'):
	f = open(filename, 'wb')
	np.save(f, matrix)
	f.close()

def import_matrix(filename='potential_matrix.npy'):
	f = open(filename, 'rb')
	matrix = np.load(f)
	f.close()
	return matrix

def compute_potential_matrix(save=True):
	start_time = t()
	x = generate_vector(x_min, x_max, Mx, Dx)
	y = generate_vector(y_min, y_max, My, Dy)
	z = generate_vector(z_min, z_max, Mz, Dz)

	mesh = create_mesh(x, y, z)
	
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
	print("Done. Final Residual: " + str(round(residual, 3)))
	computation_time = t() - start_time
	print("Total Computation Time:", computation_time, "s")
	if save:
		export_matrix(mesh)

	return mesh

def plot_data(x, y, headers, title, relzoom=1, save=False):
	fig, ax = plt.subplots(figsize=[relzoom*13.,relzoom*7.])
	ax.set_xlabel(headers["x"])
	ax.set_ylabel(headers["y"])
	ax.set_title(title)

	ax.plot(x, y, color = 'blue')

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	if save == "Y":
		figure_filename = re.sub(r'[\_\$]', r'', title ) + '.pdf'
		plt.savefig(figure_filename, bbox_inches='tight')
		f = os.path.dirname(os.path.realpath(__file__)) + "/" + figure_filename
		print("Saved figure to", f)

	plt.show()


def infinite_ppc_potential(x):
	if -d/2 <= x <= d/2:
		return V2 + (V1-V2) * (d - 2*x) / (2*d)
	elif x < -d/2:
		return V1
	else:
		return V2

def exercise_1(mesh, relzoom=1, save=False):
	mesh_center = find_center(mesh)
	axes = [
		(mesh_center[1], mesh_center[2], "O", "blue"), 
		(mesh_center[1], int((ymax-y_min)/Dy), "A", "red"), 
		(mesh_center[1], int((zmax-z_min)/Dz), "B", "green")
	]
	x = generate_vector(x_min, x_max, Mx, Dx)

	fig, ax = plt.subplots(figsize=[relzoom*13.,relzoom*7.])
	for y_index, z_index, name, color in axes:
		y = mesh[:, y_index, z_index]
		ax.plot(x, y, label="Axis $" + name + "$", marker='o', color = color)

	y = [infinite_ppc_potential(x_val) for x_val in x]
	ax.plot(x, y, label="Infinite PPC", marker='o', color = 'black')

	ax.legend()
	ax.axvspan(-d/2, d/2, alpha=0.3, color='blue')
	ax.set_title("Potential in the $x$ Direction Along Different Axes")
	ax.set_xlabel("x (cm)")
	ax.set_ylabel("Potential (V)")

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	if save:
		figure_filename = re.sub(r'[\_\$]', r'', title ) + '.pdf'
		plt.savefig(figure_filename, bbox_inches='tight')
		f = os.path.dirname(os.path.realpath(__file__)) + "/" + figure_filename
		print("Saved figure to", f)
	
	plt.show()

def exercise_2(mesh, relzoom=1, save=False):
	mesh_center = find_center(mesh)
	axes = [
		(mesh_center[1], mesh_center[2], "O", "blue"), 
		(mesh_center[1], int((ymax-y_min)/Dy), "A", "red"), 
		(mesh_center[1], int((zmax-z_min)/Dz), "B", "green")
	]
	x = generate_vector(x_min, x_max, Mx, Dx)

	fig, ax = plt.subplots(figsize=[relzoom*13.,relzoom*7.])
	for y_index, z_index, name, color in axes:
		y = mesh[:, y_index, z_index]
		ax.plot(x, y, label="Axis $" + name + "$", marker='o', color = color)

	y = [infinite_ppc_potential(x_val) for x_val in x]
	ax.plot(x, y, label="Infinite PPC", marker='o', color = 'black')

	ax.legend()
	ax.axvspan(-d/2, d/2, alpha=0.3, color='blue')
	ax.set_title("Potential in the $x$ Direction Along Different Axes")
	ax.set_xlabel("x (cm)")
	ax.set_ylabel("Potential (V)")

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()
	save = input("Do you want to save the plot? [Y/n] ")

	if save:
		figure_filename = 'Potential in the $x$ Direction Along Different Axes.pdf'
		plt.savefig(figure_filename, bbox_inches='tight')
		f = os.path.dirname(os.path.realpath(__file__)) + "/" + figure_filename
		print("Saved figure to", f)
	

# MAIN FUNCTION
if __name__=="__main__":
	print("Choose an option:")
	print("1. Compute the potential matrix from scratch")
	print("2. Import the potential matrix from a file")
	choice = input("Enter [1] or [2]: ")
	if choice == "1":
		mesh = compute_potential_matrix()
	elif choice == "2":
		mesh = import_matrix()
	else:# change this later
		print("Invalid Choice. Importing matrix.")
		mesh = import_matrix()

	print("\nChoose an option:")
	print("1. Exercise 1")
	print("2. Exercise 2")
	choice = input("Enter [1] or [2]: ")
	if choice == "1":
		exercise_1(mesh)
	elif choice == "2":
		pass
	else:
		print("Invalid Choice")



	#print_z_cut(mesh_center[2], mesh)
	#print_y_cut(mesh_center[1], mesh)
	#print_x_cut(int((x1-x_min)/Dx), mesh)
	#print_x_cut(int((x2-x_min)/Dx), mesh)
	#print_x_cut(mesh_center[0], mesh)

if __name__=="__main__2":
	mesh = import_matrix()
	exercise_2(mesh)