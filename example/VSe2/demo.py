"""Twist2D Demo."""
#%%
# +------------+
# | Usage Demo |
# +------------+
from twist2d import *

# Create an object for t2d
twist_demo = Twist2D()

# Read in the primitive cell information of each layer
twist_demo.read_primcell_of_layers('./POSCAR', 3)

# Initialize the different twisted layers 
# (for different primitive cell condition)
### For the primitive cell of VSe2 in 120 degree
m = 15
n = 8
# The key parameters for twist-2d is the mult vectors
# --> 1st layer
mult_a1p = [m, n]
mult_a2p = [-n, m-n]
twist_demo.add_layer(mult_a1p, mult_a2p, layer_dis=3)
# --> 2nd layer
mult_a1p = [n, m]
mult_a2p = [-m, n-m]
twist_demo.add_layer(mult_a1p, mult_a2p)

# ### For the primitive cell of VSe2 in 60 degree (more common)
# m = 7
# n = 8
# # --> 1st layer
# mult_a1p = [m, n]
# mult_a2p = [-n, m+n]
# twist_demo.add_layer(mult_a1p, mult_a2p, layer_dis=3)
# # --> 2nd layer
# mult_a1p = [n, m]
# mult_a2p = [-m, n+m]
# twist_demo.add_layer(mult_a1p, mult_a2p)

# Fill the cell with the layers
twist_demo.twist_layers(start_z=0.1)

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = twist_demo.calc_layers_twist_angles()
print(twisted_angles)

# Write results to the file
twist_demo.write_res_to_poscar()
