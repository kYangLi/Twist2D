"""Twist2D Demo."""
#%%
# +------------+
# | Usage Demo |
# +------------+
from twist2d import *

# Create an object for t2d
twist_demo = Twist2D()

# Initialize the different twisted layers
#  - super_a1_mult,  super_a2_mult: supercell vector a1',a2' based on a1,a2
#  - layer_dis: the layer distance of this layer to next layer, default 2A.
#  - scs_x, scs_y: supercell shift in x,y direction in angstroms, default 0A.
#  - prim_poscar: POSCAR for primitive cell of current layer, default 'POSCAR'. 
m = 6
n = 7
#--> 1st layer 
#    The 1st layer is also the base layer, which all of other layers will 
#    try to add some strain to match the 1st layer's cell constants.
super_a1_mult = [m, n]
super_a2_mult = [-n, m+n]
twist_demo.add_layer(super_a1_mult, super_a2_mult, layer_dis=3, prim_poscar="POSCAR")
#--> 2nd layer
super_a1_mult = [n, m]
super_a2_mult = [-m, n+m]
twist_demo.add_layer(super_a1_mult, super_a2_mult, prim_poscar="POSCAR")
# #--> 3rd layer
# super_a1_mult = [n, m]
# super_a2_mult = [-m, n+m]
# twist_demo.add_layer(super_a1_mult, super_a2_mult, prim_poscar="POSCAR-BN")

# Twisting the layers
#  - start_z: The lowest atom's fractional coordinates in z, default 0.1
#  - super_a3_z: The length of the c vector in z direction, default 20A.
twist_demo.twist_layers(start_z=0.1)

# Write results to the file
twist_demo.write_res_to_poscar()

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = twist_demo.calc_layers_twist_angles()
print(twisted_angles)

# PROGRAM END

#%%
# +-------------------+
# | Special condition |
# +-------------------+
from twist2d import *

# If you are twisting a bilayer graphene-like system, 
#   you can write more simply like this:

# Twist bilayer graphene-like structures
tbg_demo = TwistBGL()
tbg_demo.gen_TBGL(6, 7)
#tbg_demo.gen_TBG(m=6, n=7, prim_poscar='POSCAR', poscar_out="POSCAR.T2D.vasp", start_z=0.1, super_a3_z=20.0, layer_dis=2.0, scs_x=0.0, scs_y=0.0)

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = tbg_demo.calc_layers_twist_angles()
print(twisted_angles)

#PROGRAM END
