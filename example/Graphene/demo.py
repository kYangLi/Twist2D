"""Twist2D Demo."""
#%%
# +------------+
# | Usage Demo |
# +------------+
from twist2d import *

# Create an object for t2d
twist_demo = Twist2D()

# Initialize the different twisted layers
#  - mult_a1p,  mult_a2p: supercell vector a1',a2' based on a1,a2
#  - layer_dis: the layer distance of this layer to next layer, default 2A.
#  - scs_x, scs_y: supercell shift in x,y direction in angstroms, default 0A.
#  - prim_poscar: POSCAR for primitive cell of current layer, default 'POSCAR'. 
m = 6
n = 7
#--> 1st layer
mult_a1p = [m, n]
mult_a2p = [-n, m+n]
twist_demo.add_layer(mult_a1p, mult_a2p, layer_dis=3, prim_poscar="POSCAR")
#--> 2nd layer
mult_a1p = [n, m]
mult_a2p = [-m, n+m]
twist_demo.add_layer(mult_a1p, mult_a2p, prim_poscar="POSCAR")
# #--> 3rd layer
# mult_a1p = [n, m]
# mult_a2p = [-m, n+m]
# twist_demo.add_layer(mult_a1p, mult_a2p, prim_poscar="POSCAR-BN")

# Twisting the layers
#  - start_z: The lowest atom's fractional coordinates in z, default 0.1
#  - a3p_z: The length of the c vector in z direction, default 20A.
twist_demo.twist_layers(start_z=0.1)

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = twist_demo.calc_layers_twist_angles()
print(twisted_angles)

# Write results to the file
twist_demo.write_res_to_poscar()

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
#tbg_demo.gen_TBG(m=6, n=7, prim_poscar='POSCAR', poscar_out="POSCAR.T2D.vasp", start_z=0.1, a3p_z=20.0, layer_dis=2.0, scs_x=0.0, scs_y=0.0)

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = tbg_demo.calc_layers_twist_angles()
print(twisted_angles)

#PROGRAM END
