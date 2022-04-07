from initial_parameters import *
import numpy as np
from time import time as t
import matplotlib.pyplot as plt
import os


def create_mesh():
	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = (int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = (int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)
	
	mesh = np.ones((Mx, My, Mz)) * Vbox

	x = np.linspace(x_min, x_max, num=Mx)
	for i in range(plate1_x+1,plate2_x):
		mesh[i, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.ones((plates_y[1] - plates_y[0], plates_z[1] - plates_z[0])) * infinite_ppc_potential(x[i])
	
	mesh[plate1_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V1)
	mesh[plate2_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V2)

	return mesh

def find_center(mesh):
	x_len = len(mesh)
	y_len = len(mesh[0, :, :])
	z_len = len(mesh[0, 0, :])
	return (int(x_len/2), int(y_len/2), int(z_len/2))

def update_potential(mesh):
	plate1_x = int((x1-x_min)/Dx)
	plate2_x = int((x2-x_min)/Dx)
	plates_y = (int((ymin-y_min)/Dy), int((ymax-y_min)/Dy) + 1)
	plates_z = (int((zmin-z_min)/Dz), int((zmax-z_min)/Dz) + 1)

	new_mesh = np.ones((Mx, My, Mz)) * Vbox
	new_mesh[1:Mx-1, 1: My-1,1:Mz-1] = (mesh[0:Mx-2, 1: My-1,1:Mz-1] + \
										mesh[2:Mx, 1: My-1,1:Mz-1] + \
										mesh[1:Mx-1, 0: My-2,1:Mz-1] + \
										mesh[1:Mx-1, 2: My,1:Mz-1] + \
										mesh[1:Mx-1, 1: My-1,0:Mz-2] + \
										mesh[1:Mx-1, 1: My-1,2:Mz])/6

	new_mesh[plate1_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V1)
	new_mesh[plate2_x, plates_y[0]:plates_y[1], plates_z[0]:plates_z[1]] = np.full(shape=(int((ymax- ymin)/Dy + 1), int((zmax- zmin)/Dz + 1)), fill_value=V2)
	max_res = np.amax(np.absolute(mesh - new_mesh))

	return new_mesh, max_res

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
	# CREATE SPACE AND TIME VECTORS
	time = t()
	print("\nCreated x, y, z vectors. Time taken:", t()-time, "s")

	# CREATE INITIAL VECTOR MATRIX
	time=t()
	mesh = create_mesh()
	print("Created initial potential matrix. Time taken:", t()-time, "s")
	
	# DEFINE CONTROL PARAMETERS FOR ALGORITHM
	prev_residual = 0
	residual = 1000# (any value different to the prev_residual)
	iteration_number = 2

	# ITERATE TO UPDATE POTENTIAL
	start_time = t()
	print("\n1. Computing relaxation of potential...")
	mesh, residual = update_potential(mesh)

	while not (residual < Rtol) and residual != prev_residual:
		prev_residual = residual
		print(iteration_number, "- Computing relaxation of potential... ( Previous Residual:", round(prev_residual, 3),")")
		iteration_number += 1
		mesh, residual = update_potential(mesh)
	
	computation_time = t() - start_time
	print("Done. Final Residual: " + str(round(residual, 3)))

	# PRINT REASON FOR ENDING ALGORITHM
	if (residual < Rtol):
		reason ="Reached Tolerance"
	else:
		reason ="Reached Constant Residual"
	print("Stop Condition:", reason)

	# PRINT TOTAL COMPUTATION TIME
	print("Total Computation Time:", computation_time, "s")

	#SAVE THE MATRIX
	if save:
		export_matrix(mesh)

	return mesh

def infinite_ppc_potential(x):
	if -d/2 <= x <= d/2:
		return V2 + (V1-V2) * (d - 2*x) / (2*d)
	elif x < -d/2:
		return V1
	else:
		return V2

def exercise_1(mesh, relzoom=1):
	mesh_center = find_center(mesh)
	axes = [
		(mesh_center[1], mesh_center[2], "O", "blue"), 
		(int((ymax-y_min)/Dy), mesh_center[2], "A", "red"), 
		(mesh_center[1], int((zmax-z_min)/Dz), "B", "green")
	]
	x = np.linspace(x_min, x_max, num=Mx)

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

	save = input("Do you want to save the plot? [Y/n] ")

	if save == "Y":
		figure_filename = 'Plots/Potential in the x Direction Along Different Axes.pdf'
		plt.savefig(figure_filename)
		f = os.path.dirname(os.path.realpath(__file__)) + figure_filename
		print("Saved figure to", f)
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

def exercise_2(mesh, relzoom=1):
	mesh_center = find_center(mesh)
	x_index = mesh_center[0]
	z_index = mesh_center[2]
	
	y_axis = np.linspace(y_min, y_max, num=My)

	pot_finite = mesh[x_index, :, z_index]	
	pot_infinite = np.ones_like(pot_finite) * infinite_ppc_potential(0) 

	difference = pot_finite-pot_infinite
	percent_difference = np.absolute( difference / pot_infinite ) * 100

	# PLOT DIFFERENCE
	print("Plotting difference...")
	fig, ax = plt.subplots(figsize=[relzoom*13.,relzoom*7.])
	ax.plot(y_axis, difference, color="blue")

	ax.axvspan(ymin, ymax, alpha=0.3, color='blue')
	ax.set_title("Difference Between the Potential of the Finite and Infinite PPCs Along the OY Axis")
	ax.set_xlabel("y (cm)")
	ax.set_ylabel("Percent Difference (%)")

	save = input("Do you want to save the plot? [Y/n] ")

	if save == "Y":
		figure_filename = 'Plots/Difference Between the Potential of the Finite and Infinite PPCs Along the OY Axis.pdf'
		plt.savefig(figure_filename)
		f = os.path.dirname(os.path.realpath(__file__)) + figure_filename
		print("Saved figure to", f)
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()	

	# PLOT PERCENT DIFFERENCE
	print("Plotting percent difference...")
	fig, ax = plt.subplots(figsize=[relzoom*13.,relzoom*7.])
	ax.plot(y_axis, percent_difference, color="red")

	ax.axvspan(ymin, ymax, alpha=0.3, color='blue')
	ax.set_title("Percent Difference Between the Potential of the Finite and Infinite PPCs Along the OY Axis")
	ax.set_xlabel("y (cm)")
	ax.set_ylabel("Percent Difference (%)")

	save = input("Do you want to save the plot? [Y/n] ")

	if save == "Y":
		figure_filename = 'Plots/Percent Difference Between the Potential of the Finite and Infinite PPCs Along the OY Axis.pdf'
		plt.savefig(figure_filename)
		f = os.path.dirname(os.path.realpath(__file__)) + figure_filename
		print("Saved figure to", f)
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

	boundary = 10000000
	tolerance = 10
	for i in range(np.where(y_axis==0.)[0][0], len(y_axis)):
		if percent_difference[i] > tolerance:
			boundary = y_axis[i]
			print("The distance from the origin along the Y axis beyond which the difference is larger than", tolerance, "% is", round(boundary, 4), "cm")
			break
	
	print("The percentage difference at the left edge of the finite capacitor is", round( percent_difference[np.where(y_axis==ymin)[0][0]] , 2), "%")

def plot_3D(mesh, z_index, relzoom=1):
	x = np.linspace(x_min, x_max, num=Mx)
	y = np.linspace(y_min, y_max, num=My)
	z = np.linspace(z_min, z_max, num=Mz)
	
	X, Y = np.meshgrid(x, y)
	f = lambda x_val, y_val: mesh[np.where(x==x_val)[0][0], np.where(y==y_val)[0][0], z_index]

	f_vect = np.vectorize(f)
	Z = f_vect(X, Y)


	fig = plt.figure(figsize=(relzoom*13,relzoom*7))
	ax = plt.axes(projection="3d")
	ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='magma', edgecolor=None)
	ax.set(xlabel="x", ylabel="y", zlabel="V(x, y)", title="Potential on the $Z=" + str(z[z_index]) + "$ Plane")
		
	save = input("Do you want to save the plot? [Y/n] ")

	if save == "Y":
		figure_filename = 'Plots/Potential on the Z=' + str(z[z_index]) + ' Plane.pdf'
		plt.savefig(figure_filename)
		f = os.path.dirname(os.path.realpath(__file__)) + figure_filename
		print("Saved figure to", f)
	
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

def plot_contours(mesh, z_index, levels, relzoom=1):
	x = np.linspace(x_min, x_max, num=Mx)
	y = np.linspace(y_min, y_max, num=My)
	z = np.linspace(z_min, z_max, num=Mz)
	
	X, Y = np.meshgrid(x, y)
	f = lambda x_val, y_val: mesh[np.where(x==x_val)[0][0], np.where(y==y_val)[0][0], z_index]

	f_vect = np.vectorize(f)
	Z = f_vect(X, Y)


	fig = plt.figure(figsize=[relzoom*13.,relzoom*7.])
	ax = plt.axes()
	CS = ax.contour(X, Y, Z, levels)
	CB = fig.colorbar(CS, ticks=levels)
	plt.clabel(CS, inline=True, fontsize=7)
	
	plt.ylim([y_min,y_max])
	plt.ylim([x_min,x_max])
	ax.set(xlabel="x", ylabel="y", title="Potential levels on the $Z=" + str(z[z_index]) + "$ Plane")

	save = input("Do you want to save the plot? [Y/n] ")

	if save == "Y":
		figure_filename = 'Plots/Potential Levels on the Z=' + str(z[z_index]) + ' Plane.pdf'
		plt.savefig(figure_filename)
		f = os.path.dirname(os.path.realpath(__file__)) + figure_filename
		print("Saved figure to", f)
	
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

def exercise_3(mesh, relzoom=1):
	mesh_center = find_center(mesh)
	z_index = mesh_center[2]
	levels = [float(i) for i in range(-4, 10)]

	print("Generating 3D plot of the potential in the plane Z = 0...")
	plot_3D(mesh, z_index, relzoom=relzoom)

	print("Generating Levels plot of the potential in the plane Z = 0...")
	plot_contours(mesh, z_index, levels, relzoom=relzoom)

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
	print("3. Exercise 3")
	choice = input("Enter [1], [2] or [3]: ")
	print("")
	if choice == "1":
		exercise_1(mesh, relzoom=0.9)
	elif choice == "2":
		exercise_2(mesh)
	elif choice == "3":
		exercise_3(mesh, relzoom=0.9)
	else:
		print("Invalid Choice")