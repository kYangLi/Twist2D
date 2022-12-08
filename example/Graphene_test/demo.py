from twist2d import *

twist_demo = Twist2D()

m = 6
n = 7

super_a1_mult = [m, n]
super_a2_mult = [-n, m+n]
twist_demo.add_layer(super_a1_mult, super_a2_mult, layer_dis=3, prim_poscar="POSCAR")

super_a1_mult = [n, m]
super_a2_mult = [-m, n+m]
twist_demo.add_layer(super_a1_mult, super_a2_mult, prim_poscar="POSCAR")

twist_demo.twist_layers(start_z=0.1)

twist_demo.write_res_to_poscar()

twisted_angles = twist_demo.calc_layers_twist_angles()
print(twisted_angles)

