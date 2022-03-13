# Dimensions of the surrounding box (conductor):
Lx = 10#cm (x side of the box)
x_min = -Lx/2
x_max = Lx/2
Mx = 101
Dx = (x_max - x_min) / (Mx - 1)

Ly = 15#cm (y side of the box)
y_min = -Ly/2
y_max = Ly/2
My = 151
Dy = (y_max - y_min) / (My - 1)

Lz = 30#cm (z side of the box)
z_min = -Lz/2
z_max = Lz/2
Mz = 301
Dz = (z_max - z_min) / (Mz - 1)

# Potential of the surrounding box (conductor):
Vbox = 0#V

# Position and dimensions of the ppcs:
d = 1#cm (distance between plates)
x1 = -d/2# (x position of the first plate)
x2 = d/2# (x position of the second plate)
lz = 10#cm (length of the z side for both plates)
zmax = lz/2# (end of the z side for both plates)
zmin = -lz/2# (start of the z side for both plates)
ly = 5#cm (length of the y side for both plates)
ymax = ly/2# (end of the y side for both plates)
ymin = -ly/2# (start of the y side for both plates)

# Potential at the ppcs:
V1 = 10#V
V2 = -5#V

# Residual Tolerance:
Rtol = 0.01
