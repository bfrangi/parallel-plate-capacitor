from initial_parameters import *

# SPACE AND TIME VECTORS
def generate_mesh(vec_min, vec_max, Mv):
    v = []
    Dv = (vec_max - vec_min) / (Mv - 1)
    for k in range(Mv):
        v.append( vec_min + k*Dv )
    return np.array(v)



# MAIN FUNCTION
if __name__=="__main__":
	x = generate_mesh(x_min, x_max, Mx)
    y = generate_mesh(y_min, y_max, My)
    z = generate_mesh(z_min, z_max, Mz)