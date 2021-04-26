from twist2d import *

# Create an object for t2d
twist_demo = Twist2D()

# Read in the layers in different POSCARs here
twist_demo.read_init_poscar('./POSCAR', 2)
# twist_demo.read_init_poscar('POSCAR2', 1)

# Read in the supercell vectors in different layers
m = 6; n = 7
#--> 1st layer
mult_a1p = [12, 0]
mult_a2p = [0, 6]
twist_demo.writein_supercell_vector(mult_a1p, mult_a2p)
#--> 2nd layer
mult_a1p = [13, 1]
mult_a2p = [-1, 6]
twist_demo.writein_supercell_vector(mult_a1p, mult_a2p)

# Twist the layer!
twist_demo.gen_twisted_atoms_coord(start_z=0.1, inplane_mis='0.5,0.0', layer_dis=4)

# Write data to the file
twist_demo.write_res_poscar()

