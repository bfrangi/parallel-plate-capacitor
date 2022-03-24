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
