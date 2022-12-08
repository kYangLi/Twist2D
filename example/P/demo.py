"""Twist2D Demo."""
#%%
# +------------+
# | Usage Demo |
# +------------+
import twist2d

# Create an object for t2d
twist_demo = twist2d.Twist2D()

# Initialize the different twisted layers
super_a1_mult = [14, 0]
super_a2_mult = [0, 8]
twist_demo.add_layer(super_a1_mult, super_a2_mult, layer_dis=3)
#--> 2nd layer
super_a1_mult = [14, -1]
super_a2_mult = [1, 8]
twist_demo.add_layer(super_a1_mult, super_a2_mult)

# Twisting the layers
twist_demo.twist_layers(start_z=0.1)

# Write results to the file
twist_demo.write_res_to_poscar()

# (Optional) Calculate the twisted angles of each layer in degree 
twisted_angles = twist_demo.calc_layers_twist_angles()
print(twisted_angles)

# (Optional) Calculate the strain add (volume changes) in each layer, + for volume increasing compare to the orginal primitive cell, vice versa.
strain_vol_changes = twist_demo.calc_layers_strain()
print(strain_vol_changes)

# PROGRAM END
