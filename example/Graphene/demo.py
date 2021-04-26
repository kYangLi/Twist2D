from twist2d import *

# Create an object for t2d
twist_demo = Twist2D()

# Read in the layers in different POSCARs here
twist_demo.read_init_poscar('./POSCAR', 2)
# twist_demo.read_init_poscar('POSCAR2', 1)

# Read in the supercell vectors in different layers
m = 6; n = 7
#--> 1st layer
mult_a1p = [m, n]
mult_a2p = [-n, m+n]
twist_demo.writein_supercell_vector(mult_a1p, mult_a2p)
#--> 2nd layer
mult_a1p = [n, m]
mult_a2p = [-m, n+m]
twist_demo.writein_supercell_vector(mult_a1p, mult_a2p)
# #--> 3rd layer
# mult_a1p = [n, m]
# mult_a2p = [-m, n+m]
# twist_demo.writein_supercell_vector(mult_a1p, mult_a2p)

# Twist the layer!
twist_demo.gen_twisted_atoms_coord(start_z=0.1)

# Write data to the file
twist_demo.write_res_poscar()

