"""Module for twisting 2D materials"""
# Author: Yang Li
# Email: yangli18@mails.tsinghua.edu.cn
# Date: 2021.04.27
# Description: Module for twisting the 2D materials


import sys
import numpy as np
import math
from copy import deepcopy

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

  def _exit(self, contect='[error] Something goes wrong...'):
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

  def calc_vectors_angle(self, v1, v2):
    # Get the dot product
    v1dotv2 = np.dot(v1, v2)
    # Get the length of v1, v2
    length_v1 = np.sqrt(np.dot(v1, v1))
    length_v2 = np.sqrt(np.dot(v2, v2))
    # Calc angle
    cos_theta = v1dotv2 / (length_v1 * length_v2)
    theta = np.arccos(cos_theta)
    return theta 

  def angle_pi2degree(self, angle):
    return angle * 180 / np.pi

  # +----------------+
  # | Read in POSCAR |
  # +----------------+
  def _read_file_lines(self, filename=DEFAULT_IN_POSCAR):
    """Read in the file content to a list"""
    with open(filename) as frp:
      lines = frp.readlines()
    return lines

  def _poscar_in_dirct_mode(self, lines):
    """Check if the POSCAR file is in the 'Direct' coordinates mode."""
    direct_str = lines[7]
    if 'irect' not in direct_str:
      return False
    return True

  def _get_poscar_prim_vecs(self, lines):
    """read in the poscar primitive cell vectors"""
    length_unit = float(lines[1])
    a1_x = float(lines[2].split()[0]) * length_unit
    a1_y = float(lines[2].split()[1]) * length_unit
    a2_x = float(lines[3].split()[0]) * length_unit
    a2_y = float(lines[3].split()[1]) * length_unit
    a3_z = float(lines[4].split()[2]) * length_unit
    prim_vecs = np.array([[a1_x, a1_y], [a2_x, a2_y]])
    return prim_vecs, a3_z
  
  def _get_poscar_atoms_coord(self, lines, total_number):
    """Get the POSCAR atoms' fractional coordinates"""
    atom_coord_list = []
    for line_i in range(8, 8+total_number):
      # Get the coord of current line's atom
      coord_x = float(lines[line_i].split()[0])
      coord_y = float(lines[line_i].split()[1])
      coord_z = float(lines[line_i].split()[2])
      curr_coord = np.array([coord_x, coord_y, coord_z])
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

  def _record_primcell_info(self, primcell_info, repeat_time):
    """Record the poscar info to the poscar data list"""
    self.primcell_info_list.append(primcell_info)
    for _ in range(repeat_time-1):
      repeated_primcell_info = deepcopy(primcell_info)
      self.primcell_info_list.append(repeated_primcell_info)
    
  def read_primcell_of_layers(self, filename, repeat_time=2):
    """Read the info in the init poscar file"""
    # Read in the contant of POSCAR
    lines = self._read_file_lines(filename)
    # Check if the POSCAR in direct coordinate mode
    if not self._poscar_in_dirct_mode(lines):
      self._exit("[error] The POSCAR MUST in 'Direct' mode!")
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
    self._record_primcell_info(primcell_info, repeat_time)

  # +--------------------+
  # | Generate Supercell |
  # +--------------------+
  def _frac_coord_is_in_cell(self, coord):
    """Determine whether the fractional coordinate is in the cell"""
    frac_x = coord[0]
    frac_y = coord[1]
    return ((frac_x > 0.0) and (frac_x < 1.0) and
            (frac_y > 0.0) and (frac_y < 1.0) and 
            (not math.isclose(frac_x, 1.0, rel_tol=FLOAT_PREC)) and
            (not math.isclose(frac_y, 1.0, rel_tol=FLOAT_PREC))
            ) or math.isclose(frac_x, 0.0, rel_tol=FLOAT_PREC) \
            or math.isclose(frac_y, 0.0, rel_tol=FLOAT_PREC)

  def _get_atoms_num_in_supercell(self, super_mult_vec1, super_mult_vec2):
    """Get the atoms number in the supercell"""
    return abs(np.cross(super_mult_vec1, super_mult_vec2))

  def _get_index_boder_of_atoms(self, a1p, a2p):
    """Get the boder of the atoms indeies searching"""
    # A boder extended by 4 points: (0,0), a1p, a2p, a12p
    a12p = a1p + a2p
    a1_upper = max(0, a1p[0], a2p[0], a12p[0])
    a1_lower = min(0, a1p[0], a2p[0], a12p[0])
    a2_upper = max(0, a1p[1], a2p[1], a12p[1])
    a2_lower = min(0, a1p[1], a2p[1], a12p[1])
    a1_boder = [a1_lower, a1_upper]
    a2_boder = [a2_lower, a2_upper]
    return a1_boder, a2_boder

  def _get_supercell_vecs(self, supercell_matrix, primitive_vecs, a3p_z):
    """Get the supercell vectors"""
    # Supercell vectors in 2D
    svs_2d = np.dot(supercell_matrix, primitive_vecs)
    supercell_vecs = np.array([[svs_2d[0,0], svs_2d[0,1], 0],
                               [svs_2d[1,0], svs_2d[1,1], 0],
                               [0, 0, a3p_z]])
    return supercell_vecs
    
  def _get_supercell_shifts(self, a1_boder, a2_boder, supercell_matrix_inv):
    '''Get the supercell shift list for each sub-primitive-cell.'''
    supercell_shifts = []
    relav_coord = [0.5, 0.5]
    for a1_i in range(a1_boder[0], a1_boder[1]):
      for a2_i in range(a2_boder[0], a2_boder[1]):
        coord_x = relav_coord[0] + a1_i
        coord_y = relav_coord[1] + a2_i
        primit_coord = np.array([coord_x, coord_y])
        supercell_coord = np.dot(primit_coord, supercell_matrix_inv)
        # If the points is inside the cell or at the edge
        if self._frac_coord_is_in_cell(supercell_coord):
          supercell_shifts.append([a1_i, a2_i])
    return supercell_shifts

  def _get_supercell_atoms_coord(self, supercell_matrix_inv, a3p_z, a3_z,
                                       supercell_shifts, atom_coord_list,
                                       scell_shift_x, scell_shift_y,
                                       supercell_vecs, supercell_shift_z):
    """Get the atomic fractional coordinates in the supercell"""
    # Find the minimal frac_z of the atoms
    min_frac_z = min(np.array(atom_coord_list)[:,2])
    # Transfrom the scell shift from the Cart to Frac
    cart_vec = [scell_shift_x, scell_shift_y, 0]
    scell_shift_x, scell_shift_y, _ = \
      self.coord_cart2frac(supercell_vecs, cart_vec)
    # generate the supercell atoms
    supercell_atom_coord_list = []
    for coord in atom_coord_list:
      for shift in supercell_shifts:
        # Get the primitive cell coords
        coord_x = coord[0] + shift[0]
        coord_y = coord[1] + shift[1] 
        coord_z = (coord[2] - min_frac_z) * a3_z / a3p_z + supercell_shift_z
        primit_coord = np.array([coord_x, coord_y])
        # Get the supercell coords
        supercell_coord = np.dot(primit_coord, supercell_matrix_inv)
        supercell_coord[0] += scell_shift_x
        supercell_coord[1] += scell_shift_y
        supercell_coord = np.append(supercell_coord, coord_z)
        # Record
        supercell_atom_coord_list.append(supercell_coord)
    # Record the range of frac_z about the atoms
    min_frac_z = min(np.array(supercell_atom_coord_list)[:,2])
    max_frac_z = max(np.array(supercell_atom_coord_list)[:,2])
    frac_z_range = [min_frac_z, max_frac_z]
    return supercell_atom_coord_list, frac_z_range

  def gen_supercell(self, super_mult_vec1, super_mult_vec2, 
                          primitive_vecs, a3_z, quantities_list,
                          atom_coord_list, a3p_z=20.0, scell_shift_x=0.0,
                          scell_shift_y=0.0, supercell_shift_z=0.0):
    """Generate the suppercell"""
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
      self._get_supercell_vecs(supercell_matrix, primitive_vecs, a3p_z)
    ### Calculate supercell shifts according to supercell vector a1',a2'
    # Get the boder of the atomic index searching
    a1_boder, a2_boder = \
      self._get_index_boder_of_atoms(super_mult_vec1, super_mult_vec2)
    # Find all atoms' shift vector in the supercell
    supercell_shifts = self._get_supercell_shifts(a1_boder, a2_boder,
                                                  supercell_matrix_inv)
    # Check the atom number in supercell
    check_supercell_num = len(supercell_shifts)
    if check_supercell_num != supercell_num:
      self._exit("[error] Expect %d positions in suppercell, but find %d..." 
                  %(supercell_num, check_supercell_num))
    # Get fractional coordinates in supercell (min-z = 0.0)
    supercell_atom_coord_list, frac_z_range = \
      self._get_supercell_atoms_coord(supercell_matrix_inv, a3p_z, a3_z,
                                      supercell_shifts, atom_coord_list,
                                      scell_shift_x, scell_shift_y,
                                      supercell_vecs, supercell_shift_z)
    return supercell_vecs, supercell_quantities_list, \
           supercell_atom_coord_list, frac_z_range

  # +--------------+
  # | Twist Layers |
  # +--------------+
  def _move_atoms_to_one_cell(self, atom_coords):
    """Check and move all of the frac coordinates of atoms to one cell"""
    for coord_i, coord in enumerate(atom_coords):
      if (coord[0] > 1.0) or \
          (math.isclose(coord[0], 1.0, rel_tol=FLOAT_PREC)):
        atom_coords[coord_i][0] -= 1
      if coord[0] < 0.0:
        atom_coords[coord_i][0] += 1
      if (coord[1] > 1.0) or \
          (math.isclose(coord[1], 1.0, rel_tol=FLOAT_PREC)):
        atom_coords[coord_i][1] -= 1
      if coord[1] < 0.0:
        atom_coords[coord_i][1] += 1
      if not self._frac_coord_is_in_cell(atom_coords[coord_i]):
        print("[error] Vector error: [%f, %f, %f] ..." 
                %(coord[0], coord[1], coord[2]))
    return atom_coords

  def init_twisted_layers(self, mult_a1p, mult_a2p, layer_dis=2, 
                                scs_x=0.0, scs_y=0.0):
    """Read in the the layers' parameters in supercell"""
    curr_supercell_info = {"super_mult_a1"  : np.array(mult_a1p),
                           "super_mult_a2"  : np.array(mult_a2p),
                           "layer_distance" : layer_dis,
                           "scell_shift_x"  : scs_x,
                           "scell_shift_y"  : scs_y,}
    self.supercell_info_list.append(curr_supercell_info)

  def fill_twisted_layers_cell(self, start_z=0.0, a3p_z=20.0):
    """Generate the twisted layers atoms fractional coordinates."""
    # Check if supercell info is complete
    if len(self.supercell_info_list) != len(self.primcell_info_list):
      self._exit("[error] supercell info not ready...")
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
      # Generate the suppercell
      supercell_vecs, supercell_quantities_list, supercell_atom_coord_list, \
        frac_z_range = self.gen_supercell(super_mult_a1, super_mult_a2, 
                                          primitive_vecs, a3_z, quantities_list,
                                          atom_coord_list, a3p_z,
                                          scell_shift_x, scell_shift_y,
                                          supercell_shift_z)
      # Update the supercell shift z
      supercell_shift_z = frac_z_range[1] + layer_dis/a3p_z
      if supercell_shift_z >= 1.0:
        self._exit("[error] Coordinate z is out of range, pls reduce the layer distance or increase the cell length of z.")
      # Recheck and move all of atoms to one supercell
      supercell_atom_coord_list = \
        self._move_atoms_to_one_cell(supercell_atom_coord_list)
      # Record the current rotated supercell info
      self.supercell_info_list[i]["supercell_vecs"] = supercell_vecs
      self.supercell_info_list[i]["supercell_quantities_list"] = \
        supercell_quantities_list
      self.supercell_info_list[i]["supercell_atom_coord_list"] = \
        supercell_atom_coord_list

  def calc_layers_twist_angles(self):
    twisted_angles = []
    ref_vec_a1 = self.supercell_info_list[0]["supercell_vecs"][0]
    for supercell_info in self.supercell_info_list:
      supercell_vec_a1 = supercell_info["supercell_vecs"][0]
      theta = self.calc_vectors_angle(ref_vec_a1, supercell_vec_a1)
      theta = self.angle_pi2degree(theta)
      twisted_angles.append(theta)
    return twisted_angles

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
        "    % 4.8f    % 4.8f    % 4.8f" %(vec[0], vec[1], vec[2]))
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
  def init_graphenelike_slayers(self, m, n, layer_dis, scs_x, scs_y):
    """Write in the graphene like supercell vectors"""
    # 1st layer
    mult_a1p = [m, n]
    mult_a2p = [-n, m+n]
    self.init_twisted_layers(mult_a1p, mult_a2p, layer_dis, scs_x, scs_y)
    # 2nd layer
    mult_a1p = [n, m]
    mult_a2p = [-m, n+m]
    self.init_twisted_layers(mult_a1p, mult_a2p, layer_dis, scs_x, scs_y)

  def gen_TBG(self, m, n,
              poscar_init=DEFAULT_IN_POSCAR, poscar_out=DEFAULT_OUT_POSCAR,
              start_z=0.1, a3p_z=20.0, layer_dis=2.0, scs_x=0.0, scs_y=0.0):
    """Generate the twisted bilayer graphene(TBG) system."""
    self.read_primcell_of_layers(poscar_init, repeat_time=2)
    self.init_graphenelike_slayers(m, n, layer_dis, scs_x, scs_y)
    self.fill_twisted_layers_cell(start_z, a3p_z)
    # Update the out POSCAR name
    if poscar_out == DEFAULT_OUT_POSCAR:
      poscar_out = 'POSCAR.T2D-%dx%d.vasp' %(m, n)
    # Save the data to out POSCAR
    self.write_res_to_poscar(poscar_out)
      