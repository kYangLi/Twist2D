"""Module for twisting 2D materials"""
# Author: Yang Li
# Email: yangli18@mails.tsinghua.edu.cn
# Date: 2021.04.27
# Description: 
# Module for twisting the 2D materials,
#   Special thanks for 
#   guoxm teach me the wonderful integer supercell generation method


import sys
import os
import numpy as np


# Necessary Constants
FLOAT_PREC = 1e-6
DEFAULT_IN_POSCAR = 'POSCAR'
DEFAULT_OUT_POSCAR = 'POSCAR.T2D.vasp'


############################
### Class for 2d twisted ###
############################
class Twist2D():
  # +-------------+
  # | Sys Setting |
  # +-------------+
  def __init__(self):
    self.primcell_info_list = []
    self.supercell_info_list = []


  def _exit(self, contect='[error] Unknown: Something goes wrong...'):
    """exit the program with some error msg."""
    print(contect)
    sys.exit(1)


  # +------------+
  # | Misc Tools |
  # +------------+
  def coord_cart2frac(self, cell_vecs, cart_vec):
    """Transfrom the cart coords to frac coords"""
    cell_vecs_inv = np.linalg.inv(cell_vecs)
    frac_vec = np.dot(cart_vec, cell_vecs_inv)
    return frac_vec


  def float_eq(self, f1, f2, prec=FLOAT_PREC):
    """float equal"""
    return abs(f1 - f2) < prec


  def cross_a2d(self, v1, v2):
    """Calculate the value of cross mult between two 2d array"""
    return v1[0]*v2[1] - v1[1]*v2[0]

  
  def abs_cross_a2d(self, v1, v2):
    """Calculate the abs value of cross mult between two 2d array"""
    return abs(self.cross_a2d(v1, v2))


  def calc_vectors_angle(self, v1, v2):
    """calculate the angle between two vectors"""
    # Get the dot product
    v1dotv2 = np.dot(v1, v2)
    # Get the length of v1, v2
    length_v1 = np.sqrt(np.dot(v1, v1))
    length_v2 = np.sqrt(np.dot(v2, v2))
    # Get cos_theta
    cos_theta = v1dotv2 / (length_v1 * length_v2)
    # In the case cos(theta) little larger than 1.0
    if self.float_eq(cos_theta, 1.0):
      cos_theta = 1.0
    # Get theta
    theta = np.arccos(cos_theta)
    return theta 


  def calc_vectors_angle_wsign(self, v1, v2):
    """calculate the angle between two vectors with the right hand rule"""
    # Get the angle
    theta = self.calc_vectors_angle(v1, v2)
    # Get the sign
    sign = np.sign(self.cross_a2d(v1, v2))
    return sign * theta
    

  def angle_pi2degree(self, angle):
    """angle pi to degree"""
    return angle * 180 / np.pi


  # +----------------+
  # | Read in POSCAR |
  # +----------------+
  def _read_file_lines(self, filename=DEFAULT_IN_POSCAR):
    """Read in the file content to a list"""
    if not os.path.isfile(filename):
      self._exit("[error] Primitive POSCAR: file '%s' not found." %filename)
    with open(filename) as frp:
      lines = frp.readlines()
    return lines


  def _poscar_in_direct_mode(self, lines):
    """Check if the POSCAR file is in the 'Direct' coordinates mode."""
    direct_str = lines[7]
    return ('direct' in direct_str.lower())


  def _get_poscar_prim_vecs(self, lines):
    """read in the poscar primitive cell vectors"""
    length_unit = float(lines[1])
    a1_x = float(lines[2].split()[0]) * length_unit
    a1_y = float(lines[2].split()[1]) * length_unit
    a1_z = float(lines[2].split()[2]) * length_unit
    a2_x = float(lines[3].split()[0]) * length_unit
    a2_y = float(lines[3].split()[1]) * length_unit
    a2_z = float(lines[3].split()[2]) * length_unit
    a3_x = float(lines[4].split()[0]) * length_unit
    a3_y = float(lines[4].split()[1]) * length_unit
    a3_z = float(lines[4].split()[2]) * length_unit
    prim_vecs = np.array([[a1_x, a1_y], [a2_x, a2_y]])
    if not (self.float_eq(a1_z, 0.0) and
            self.float_eq(a2_z, 0.0) and
            self.float_eq(a3_x, 0.0) and
            self.float_eq(a3_y, 0.0)):
      self._exit("[error] Primitive POSCAR: The c axis must in z direction!")
    return prim_vecs, a3_z
  

  def _get_poscar_atoms_coord(self, lines, total_number):
    """Get the POSCAR atoms' fractional coordinates"""
    atom_coord_list = []
    for line_i in range(8, 8+total_number):
      # Get the coord of current line's atom
      coord_a = float(lines[line_i].split()[0])
      coord_b = float(lines[line_i].split()[1])
      coord_c = float(lines[line_i].split()[2])
      curr_coord = np.array([coord_a, coord_b, coord_c])
      atom_coord_list.append(curr_coord)
    return atom_coord_list


  def _get_poscar_atoms_info(self, lines):
    """Get the atoms's info in the poscar"""
    # Get the atoms' symbols
    elements_list = lines[5].split()
    # Get the atoms' number
    quantities_list = list(map(int, lines[6].split()))
    # Read in the atoms coords
    atom_coord_list = self._get_poscar_atoms_coord(lines, sum(quantities_list))
    return elements_list, quantities_list, atom_coord_list


  def _record_primcell_info(self, primcell_info):
    """Record the poscar info to the poscar data list"""
    self.primcell_info_list.append(primcell_info)


  def read_primcell_of_layers(self, filename):
    """Read the info in the init poscar file"""
    # Read in the contant of POSCAR
    lines = self._read_file_lines(filename)
    # Check if the POSCAR in direct coordinate mode
    if not self._poscar_in_direct_mode(lines):
      self._exit("[error] Primitive POSCAR: the atoms' coordinates MUST in 'Direct' mode!")
    # Get the 2D vector of a1 a2
    prim_vecs, a3_z = self._get_poscar_prim_vecs(lines)
    # Get the atoms' info
    elements_list, quantities_list, atom_coord_list = \
      self._get_poscar_atoms_info(lines)
    # Create and record the result
    primcell_info = {"prim_vecs"   : prim_vecs,
                     "a3_z"        : a3_z,
                     "elements"    : elements_list,
                     "quantities"  : quantities_list,
                     "atom_coords" : atom_coord_list}
    self._record_primcell_info(primcell_info)


  # +--------------------+
  # | Generate Supercell |
  # +--------------------+
  def _get_atoms_num_in_supercell(self, super_mult_vec1, super_mult_vec2):
    """Get the atoms number in the supercell"""
    return self.abs_cross_a2d(super_mult_vec1, super_mult_vec2)


  def _get_index_boder_of_atoms(self, a1p, a2p):
    """Get the boder of the atoms indeies searching"""
    # A boder extended by 4 points: (0,0), a1p, a2p, a12p
    a12p = a1p + a2p
    a1_upper = max(0, a1p[0], a2p[0], a12p[0])
    a1_lower = min(0, a1p[0], a2p[0], a12p[0])
    a2_upper = max(0, a1p[1], a2p[1], a12p[1])
    a2_lower = min(0, a1p[1], a2p[1], a12p[1])
    a1_boder = [a1_lower, a1_upper+1]
    a2_boder = [a2_lower, a2_upper+1]
    return a1_boder, a2_boder


  def _get_supercell_vecs(self, supercell_matrix, primitive_vecs, super_a3_z):
    """Get the supercell vectors"""
    # Supercell vectors in 2D
    svs_2d = np.dot(supercell_matrix, primitive_vecs)
    supercell_vecs = np.array([[svs_2d[0,0], svs_2d[0,1], 0],
                               [svs_2d[1,0], svs_2d[1,1], 0],
                               [0, 0, super_a3_z]])
    return supercell_vecs
    

  def _get_atoms_cell_shifts(self, a1_boder, a2_boder, 
                                   supercell_a1p, supercell_a2p):
    '''Get the atoms cell shifts in the supercell for each sub-primitive-cell. new_position_in_supercell = cell_shifts + atom_pos_in_primcell'''
    atoms_cell_shifts = []
    total_area = self.cross_a2d(supercell_a1p, supercell_a2p)
    for a1_i in range(a1_boder[0], a1_boder[1]):
      for a2_i in range(a2_boder[0], a2_boder[1]):
        shift_a1a2 = np.array([a1_i, a2_i])
        # !!! KEY code !!!
        supercell_a1_int = \
          self.cross_a2d(shift_a1a2, supercell_a2p) // total_area
        supercell_a2_int = \
          self.cross_a2d(supercell_a1p, shift_a1a2) // total_area
        if (supercell_a1_int == 0) and (supercell_a2_int == 0):
          atoms_cell_shifts.append(shift_a1a2)
    return atoms_cell_shifts


  def _get_coords_z_range(self, coord_list):
    """Get the range of atoms' fractional coordinate z"""
    min_z = min(np.array(coord_list)[:,2])
    max_z = max(np.array(coord_list)[:,2])
    return min_z, max_z


  def _get_supercell_atoms_coord(self, supercell_matrix_inv, super_a3_z, a3_z,
                                       atoms_cell_shifts, atom_coord_list,
                                       scell_shift_x, scell_shift_y,
                                       supercell_vecs, supercell_shift_z):
    """Get the atomic fractional coordinates in the supercell"""
    # Find the minimal frac_z of the atoms
    min_frac_z, _ = self._get_coords_z_range(atom_coord_list)
    # Transfrom the scell shift from the Cart to Frac
    cart_vec = [scell_shift_x, scell_shift_y, 0]
    scell_shift_x, scell_shift_y, _ = \
      self.coord_cart2frac(supercell_vecs, cart_vec)
    # generate the supercell atoms
    supercell_atom_coord_list = []
    for coord in atom_coord_list:
      for shift in atoms_cell_shifts:
        # Get the primitive cell coords
        coord_a = coord[0] + shift[0]
        coord_b = coord[1] + shift[1] 
        coord_c = (coord[2] - min_frac_z) * a3_z / super_a3_z + supercell_shift_z
        primit_coord = np.array([coord_a, coord_b])
        # Get the supercell coords
        supercell_coord = np.dot(primit_coord, supercell_matrix_inv)
        supercell_coord[0] += scell_shift_x
        supercell_coord[1] += scell_shift_y
        supercell_coord = np.append(supercell_coord, coord_c)
        # Record
        supercell_atom_coord_list.append(supercell_coord)
    # Record the range of frac_z about the atoms
    frac_z_range = self._get_coords_z_range(supercell_atom_coord_list)
    return supercell_atom_coord_list, frac_z_range


  def _check_supercell_cellnum(self, atoms_cell_shifts, supercell_num):
    """Check the supercell atoms' number"""
    check_supercell_num = len(atoms_cell_shifts)
    if check_supercell_num != supercell_num:
      self._exit("[error] Supercell generation: expect %d primitive cell in supercell, but find %d..." 
                 %(supercell_num, check_supercell_num))


  def gen_supercell(self, super_mult_vec1, super_mult_vec2, 
                          primitive_vecs, a3_z, quantities_list,
                          atom_coord_list, super_a3_z=20.0, scell_shift_x=0.0,
                          scell_shift_y=0.0, supercell_shift_z=0.0):
    """Generate the supercell"""
    # Supercell number
    supercell_num = self._get_atoms_num_in_supercell(super_mult_vec1,
                                                     super_mult_vec2)
    # Supercell number list
    supercell_quantities_list = [val*supercell_num for val in quantities_list]
    # Supercell transfrom matrix
    supercell_matrix = np.array([super_mult_vec1, super_mult_vec2])
    supercell_matrix_inv = np.linalg.inv(supercell_matrix)
    # Supercell vectors 
    supercell_vecs = \
      self._get_supercell_vecs(supercell_matrix, primitive_vecs, super_a3_z)
    # Calculate supercell shifts according to supercell vector a1',a2'
    #--> Get the boder of the atomic index searching
    a1_boder, a2_boder = \
      self._get_index_boder_of_atoms(super_mult_vec1, super_mult_vec2)
    #--> Find all atoms' shift vector in the supercell
    atoms_cell_shifts = self._get_atoms_cell_shifts(a1_boder, a2_boder,
                                                    super_mult_vec1,
                                                    super_mult_vec2)
    #--> Check the atom number in supercell
    self._check_supercell_cellnum(atoms_cell_shifts, supercell_num)
    # Get fractional coordinates in supercell (min-z = 0.0)
    supercell_atom_coord_list, frac_z_range = \
      self._get_supercell_atoms_coord(supercell_matrix_inv, super_a3_z, a3_z,
                                      atoms_cell_shifts, atom_coord_list,
                                      scell_shift_x, scell_shift_y,
                                      supercell_vecs, supercell_shift_z)
    return supercell_vecs, supercell_quantities_list, \
           supercell_atom_coord_list, frac_z_range


  # +--------------+
  # | Twist Layers |
  # +--------------+
  def add_layer(self, super_a1_mult, super_a2_mult, 
                      layer_dis=2, scs_x=0.0, scs_y=0,
                      prim_poscar=DEFAULT_IN_POSCAR):
    """Read in the the layers' parameters in supercell"""
    # Read in the primitive cell info
    self.read_primcell_of_layers(prim_poscar)
    # Read in the supercell info
    curr_supercell_info = {"super_mult_a1"  : np.array(super_a1_mult),
                           "super_mult_a2"  : np.array(super_a2_mult),
                           "layer_distance" : layer_dis,
                           "scell_shift_x"  : scs_x,
                           "scell_shift_y"  : scs_y,}
    self.supercell_info_list.append(curr_supercell_info)
  
  
  def _check_primvecs_in_layers(self):
    """Check the primitive cell vectors in different layers"""
    # Get the reference vector of primitive cell
    ref_primcell_vecs = self.primcell_info_list[0]["prim_vecs"]
    ref_p_a1 = ref_primcell_vecs[0]
    ref_p_a2 = ref_primcell_vecs[1]
    # Check each primitive cell vectors
    for i, primcell_info in enumerate(self.primcell_info_list):
      layer_i = i + 1
      prim_vecs = primcell_info["prim_vecs"]
      p_a1 = prim_vecs[0]
      p_a2 = prim_vecs[1]
      theta_1 = self.calc_vectors_angle_wsign(ref_p_a1, p_a1)
      theta_2 = self.calc_vectors_angle_wsign(ref_p_a2, p_a2)
      # Discuss the different cases:
      if theta_1 * theta_2 < 0.0:
        print("[warning] Layer %d: the chirality of this layer's primitive cell vectors do not agree with the 1st layer's. Be careful when pick the super_a1,2_mult vector of this layer." %layer_i)
      if not self.float_eq(abs(theta_1), abs(theta_2)):
        print("[warning] Layer %d: the angle between primitive cell vectors a1,a2 in this layer do not agree with the 1st layer's. Be careful when pick the super_a1,2_mult vector of this layer." %layer_i)


  def twist_layers(self, start_z=0.0, super_a3_z=20.0):
    """Generate the twisted layers atoms fractional coordinates."""
    # Check the layers firstly
    self._check_primvecs_in_layers()
    # For each twist layers
    supercell_shift_z = start_z
    for i, supercell_info in enumerate(self.supercell_info_list):
      primcell_info = self.primcell_info_list[i]
      # Read in necessary supercell information
      super_mult_a1 = supercell_info["super_mult_a1"]
      super_mult_a2 = supercell_info["super_mult_a2"]
      layer_dis = supercell_info["layer_distance"]
      scell_shift_x = supercell_info["scell_shift_x"]
      scell_shift_y = supercell_info["scell_shift_y"]
      primitive_vecs = primcell_info["prim_vecs"]
      a3_z = primcell_info["a3_z"]
      quantities_list = primcell_info["quantities"]
      atom_coord_list = primcell_info["atom_coords"]
      # Generate the supercell
      supercell_vecs, supercell_quantities_list, \
        supercell_atom_coord_list, frac_z_range = \
        self.gen_supercell(super_mult_a1, super_mult_a2, primitive_vecs, 
                           a3_z, quantities_list, atom_coord_list, super_a3_z,
                           scell_shift_x, scell_shift_y, supercell_shift_z)
      # Update the supercell shift z
      supercell_shift_z = frac_z_range[1] + (layer_dis / super_a3_z)
      if supercell_shift_z > 1.0:
        self._exit("[error] Twisting layers: coordinate z is out of range, pls reduce the layer distance or increase the cell length of z.")
      # Record the current rotated supercell info
      self.supercell_info_list[i]["supercell_vecs"] = supercell_vecs
      self.supercell_info_list[i]["supercell_quantities_list"] = \
        supercell_quantities_list
      self.supercell_info_list[i]["supercell_atom_coord_list"] = \
        supercell_atom_coord_list


  def calc_layers_twist_angles(self):
    """Calculate the layers twisted angles"""
    twisted_angles = []
    ref_vec_a1 = self.supercell_info_list[0]["supercell_vecs"][0]
    for supercell_info in self.supercell_info_list:
      supercell_vec_a1 = supercell_info["supercell_vecs"][0]
      theta = self.calc_vectors_angle(ref_vec_a1, supercell_vec_a1)
      theta = self.angle_pi2degree(theta)
      twisted_angles.append(theta)
    return twisted_angles


  def calc_layers_strain(self):
    """calculate the strain added in each layer (volume changes)"""
    layers_vol_changes = []
    ref_svec = self.supercell_info_list[0]["supercell_vecs"]
    ref_area = self.abs_cross_a2d(ref_svec[0][:2], ref_svec[1][:2])
    for scell_info in self.supercell_info_list:
      svec = scell_info["supercell_vecs"]
      area = self.abs_cross_a2d(svec[0][:2], svec[1][:2])
      vol_change = (ref_area / area) - 1
      layers_vol_changes.append(vol_change)
    return layers_vol_changes



  # +---------------+
  # | POSCAR Output |
  # +---------------+
  def _find_element_in_2dlist(self, list_2d, targ_ele):
    """Find the element index in a 2d list"""
    res_indeies = []
    for i, list_1d in enumerate(list_2d):
      for j, curr_ele in enumerate(list_1d):
        if targ_ele == curr_ele:
          res_indeies.append([i, j])
    return res_indeies


  def _combine_poscars(self):
    """Combine the elements list in different layers' POSCAR"""
    # Get symbol list list
    elements_list_list = [
      self.primcell_info_list[i]["elements"] 
      for i in range(len(self.primcell_info_list))
    ]
    # Get atoms number list list
    quantities_list_list = [
      self.supercell_info_list[i]["supercell_quantities_list"]
      for i in range(len(self.supercell_info_list))
    ]
    # Get atoms coordinates
    atom_coords_list_list = [
      self.supercell_info_list[i]["supercell_atom_coord_list"]
      for i in range(len(self.supercell_info_list))
    ]
    # Output the orginzed POSCAR info
    org_elements_list = []
    org_quantities_list = []
    org_atom_coords_list = []
    for poscar_i, elements_list in enumerate(elements_list_list):
      for sym_i, sym in enumerate(elements_list):
        # If this symbol has already be recorded
        if sym in org_elements_list:
          continue
        # Find if there are any similar elements in other poscars
        sym_indeies = self._find_element_in_2dlist(elements_list_list, sym)
        sym_num = 0
        for row_i, col_i in sym_indeies:
          curr_sym_num = quantities_list_list[row_i][col_i]
          sym_num += curr_sym_num
          start_i = sum(quantities_list_list[row_i][:col_i])
          org_atom_coords_list += \
            atom_coords_list_list[row_i][start_i:start_i+curr_sym_num]
        # Record the symbol and number to the list
        org_elements_list.append(sym)
        org_quantities_list.append(sym_num)
    return org_elements_list, org_quantities_list, org_atom_coords_list


  def write_res_to_poscar(self, filename=DEFAULT_OUT_POSCAR):
    """Write the output POSCAR"""
    # Generate POSCAR
    out_poscar_lines = []
    # 1st comment line
    out_poscar_lines.append("Generated by Twist2D")
    out_poscar_lines.append("    1.0")
    # Cell vectors
    supercell_vecs = self.supercell_info_list[0]["supercell_vecs"]
    for vec in supercell_vecs:
      out_poscar_lines.append(
        "    %13s    %13s    %13s" %("%4.8f"%vec[0], 
                                     "%4.8f"%vec[1], 
                                     "%4.8f"%vec[2]))
    # Atoms symbol list
    org_elements_list, org_quantities_list, org_atom_coords_list = \
      self._combine_poscars()
    org_elements_list_str = '  '.join(org_elements_list)
    out_poscar_lines.append("    %s" %(org_elements_list_str))
    # Atoms number list
    org_quantities_list = [str(val) for val in org_quantities_list]
    org_quantities_list_str = '  '.join(org_quantities_list)
    out_poscar_lines.append("    %s" %(org_quantities_list_str))
    # Direct line
    out_poscar_lines.append("Direct")
    # Atoms coords
    for coord in org_atom_coords_list:
      out_poscar_lines.append(
        "    % .8f    % .8f    % .8f" %(coord[0], coord[1], coord[2]))
    # Add \n in each end of line
    for line_i, _ in enumerate(out_poscar_lines):
      out_poscar_lines[line_i] += '\n'
    # Write the POSCAR to file
    print("[do] Write the twisted structure to %s ..." %filename)
    with open(filename, 'w') as fwp:
      fwp.writelines(out_poscar_lines)


######################
### Special System ###
######################
class TwistBGL(Twist2D):
  def _check_angle(self, angle):
    # Get the vector of primitive cell
    for i, primcell_info in enumerate(self.primcell_info_list):
      layer_i = i + 1
      prim_vecs = primcell_info["prim_vecs"]
      p_a1 = prim_vecs[0]
      p_a2 = prim_vecs[1]
      # Calculate the angle
      phi = self.calc_vectors_angle(p_a1, p_a2)
      phi = self.angle_pi2degree(phi)
      # Compare to the target angle
      if not self.float_eq(phi, angle, prec=1E-3):
        self._exit("[error] Layer %d: the primitive cell vectors' angle must be 60 degree, current is %f." %(layer_i, phi))


  def add_graphenelike_layers(self, m, n, layer_dis, scs_x, scs_y, prim_poscar):
    """Write in the graphene like supercell vectors"""
    # 1st layer
    super_a1_mult = [m, n]
    super_a2_mult = [-n, m+n]
    self.add_layer(super_a1_mult, super_a2_mult, layer_dis, scs_x, scs_y, prim_poscar)
    # 2nd layer
    super_a1_mult = [n, m]
    super_a2_mult = [-m, n+m]
    self.add_layer(super_a1_mult, super_a2_mult, layer_dis, scs_x, scs_y, prim_poscar)
    # Check if the primitive vector angle is 60 degree
    self._check_angle(60)


  def gen_TBGL(self, m, n,
                     prim_poscar=DEFAULT_IN_POSCAR, 
                     poscar_out=DEFAULT_OUT_POSCAR,
                     start_z=0.1, super_a3_z=20.0, layer_dis=2.0, 
                     scs_x=0.0, scs_y=0.0):
    """Generate the twisted bilayer graphene(TBG) system."""
    self.add_graphenelike_layers(m, n, layer_dis, scs_x, scs_y, prim_poscar)
    self.twist_layers(start_z, super_a3_z)
    # Update the out POSCAR name
    if poscar_out == DEFAULT_OUT_POSCAR:
      poscar_out = 'POSCAR.T2D-%dx%d.vasp' %(m, n)
    # Save the data to out POSCAR
    self.write_res_to_poscar(poscar_out)
      