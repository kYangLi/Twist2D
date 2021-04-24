""" Python code for twist the bilayer graphene """
# Author: yangli18 
# Date: 2021.04.23
# Description: 
#


import os
import sys
import numpy as np
import math
import argparse

FLOAT_PREC = 1e-6

def get_command_line_input():
  """Read in the command line input"""
  parser = argparse.ArgumentParser("Basic parameters for TBG construct.")
  parser.add_argument('-m', '--num-vec1', dest='m', 
                      default=6, type=int,
                      help='The less(n-1) one supercell length.')
  parser.add_argument('-n', '--num-vec2', dest='n', 
                      default=7, type=int,
                      help='The large(m+1) one supercell length.')
  parser.add_argument('-z', '--cell-z', dest='len_z', 
                      default=20.0, type=float,
                      help='The length of z in angstroms.')
  parser.add_argument('-d', '--lay-dis', dest='layer_dis', 
                      default=2.0, type=float,
                      help='The distance between two layers of graphene in angstroms.')
  parser.add_argument('-s', '--start-z', dest='start_z', 
                      default=0.1, type=float,
                      help='The distance between two layers of graphene in angstroms.')
  parser.add_argument('-p', '--in-plane-mis', dest='inplane_mis', 
                      default='0.0,0.0', type=str,
                      help='In plane mis-match about two layers in primitive cell fraction coordinates. input two numbers split by "," without any blank, corresponding to frac_x,frac_y. default is coinciding at origin.')
  parser.add_argument('-i', '--init-file', dest='init_poscar', 
                      default="POSCAR", type=str,
                      help='The initial POSCAR file.')
  parser.add_argument('-o', '--out-file', dest='out_poscar', 
                      default="POSCAR.DEFAULT", type=str,
                      help='The output file name')
                    
  args = parser.parse_args()
  comline_args = {"m"           : args.m,
                  "n"           : args.n,
                  "len_z"       : args.len_z,
                  "layer_dis"   : args.layer_dis,
                  "start_z"     : args.start_z,
                  "inplane_mis" : args.inplane_mis,
                  "init_poscar" : args.init_poscar,
                  "out_poscar"  : args.out_poscar,}
  return comline_args


def read_init_poscar(init_poscar):
  """Read the info in the init poscar file"""
  # Read in the contant of POSCAR
  with open(init_poscar) as frp:
    lines = frp.readlines()
  # Get the 2D vector of a1 a2
  length_unit = float(lines[1])
  a1_x = float(lines[2].split()[0]) * length_unit
  a1_y = float(lines[2].split()[1]) * length_unit
  a2_x = float(lines[3].split()[0]) * length_unit
  a2_y = float(lines[3].split()[1]) * length_unit
  primitive_vecs = np.array([[a1_x, a1_y], [a2_x, a2_y]])
  # Get the atoms' symbols
  symbols_list = lines[5].split()
  # Get the atoms' number
  number_list = list(map(int, lines[6].split()))
  total_number = sum(number_list)
  # Check if is the direct
  direct_str = lines[7]
  if 'irect' not in direct_str:
    print("[error] The POSCAR MUST in the 'Direct' mode!")
    sys.exit(1)
  # Read in the atoms coords
  atom_coord_list = []
  for line_i in range(8, 8+total_number):
    # Get the coord of current line's atom
    coord_x = float(lines[line_i].split()[0])
    coord_y = float(lines[line_i].split()[1])
    curr_coord = np.array([coord_x, coord_y])
    atom_coord_list.append(curr_coord)
  # RETURN
  return primitive_vecs, symbols_list, number_list, atom_coord_list


def is_frac_coord_in_cell(coord):
  """Determine whether the fractional coordinates are in the cell"""
  frac_x = coord[0]
  frac_y = coord[1]
  return ((frac_x > 0.0) and (frac_x < 1.0) and
          (frac_y > 0.0) and (frac_y < 1.0) and 
          (not math.isclose(frac_x, 1.0, rel_tol=FLOAT_PREC)) and
          (not math.isclose(frac_y, 1.0, rel_tol=FLOAT_PREC))) \
         or math.isclose(frac_x, 0.0, rel_tol=FLOAT_PREC) \
         or math.isclose(frac_y, 0.0, rel_tol=FLOAT_PREC)


def gen_supercell(super_mult_vec1, super_mult_vec2, primitive_vecs,
                  atom_coord_list, number_list, 
                  incell_shift_x=0.0, incell_shift_y=0.0):
  """Generate the supercell"""
  ### Prepare for the supercell generation
  smv1_a1 = super_mult_vec1[0]
  smv1_a2 = super_mult_vec1[1]
  smv2_a1 = super_mult_vec2[0]
  smv2_a2 = super_mult_vec2[1]
  mult_a1_list = [smv1_a1, smv2_a1, 0, smv1_a1+smv2_a1]
  mult_a2_list = [smv1_a2, smv2_a2, 0, smv1_a2+smv2_a2]
  supercell_num = abs(np.cross(super_mult_vec1, super_mult_vec2))
  #supercell_num = m*m + m*n + n*n
  supercell_matrix = np.array([super_mult_vec1, super_mult_vec2])
  supercell_inv_matrix = np.linalg.inv(supercell_matrix)
  #supercell_inv_matrix = (1/(m*m + n*n +m*n)) * np.array([[m+n, -n], [n, m]])
  # Update the number list of atoms
  supercell_number_list = [val*supercell_num for val in number_list]
  # Update the cell vectors
  supercell_vecs = np.dot(supercell_matrix, primitive_vecs)
  # Update the atoms coords set:
  # Calculate supercell shifts according to 
  #   multiple on primitive cell vector a1,a2
  supercell_shift_list = []
  relav_coord = [0.5, 0.5]
  for a1_i in range(min(mult_a1_list), max(mult_a1_list)):
    for a2_i in range(min(mult_a2_list), max(mult_a2_list)):
      coord_x = relav_coord[0] + a1_i
      coord_y = relav_coord[1] + a2_i
      primit_coord = np.array([coord_x, coord_y])
      supercell_coord = np.dot(primit_coord, supercell_inv_matrix)
      # If the points is inside the cell or at the edge
      if is_frac_coord_in_cell(supercell_coord):
        supercell_shift_list.append([a1_i, a2_i])
  check_supercell_num = len(supercell_shift_list)
  # Check the shift vectors quantity
  if check_supercell_num != supercell_num:
    print("[error] The atoms' quantity in the shift list do not agree with what we expected!")
    print("[info] Total shift vector: %d" %(check_supercell_num))
    print("[info] But we expected: |mult_vec1 x mult_vec2| = %d" 
          %(supercell_num))
    sys.exit(1)
  # Add atoms
  supercell_atom_coord_list = []
  for coord in atom_coord_list:
    for shift in supercell_shift_list:
      coord_x = coord[0] + incell_shift_x + shift[0]
      coord_y = coord[1] + incell_shift_y + shift[1] 
      primit_coord = np.array([coord_x, coord_y])
      supercell_coord = np.dot(primit_coord, supercell_inv_matrix)
      supercell_atom_coord_list.append(supercell_coord)
  # RETURN
  return supercell_number_list, supercell_vecs, supercell_atom_coord_list


def gen_TBG_coords(atom_coord_list, number_list, primitive_vecs, comline_args):
  """Generate the TBG atoms coords"""
  # Read in parameters
  m = comline_args["m"]
  n = comline_args["n"]
  start_z = comline_args["start_z"]
  len_z = comline_args["len_z"]
  layer_dis = comline_args["layer_dis"]
  inplance_mis_str = comline_args["inplane_mis"]
  mis_frac_x = float(inplance_mis_str.split(',')[0])
  mis_frac_y = float(inplance_mis_str.split(',')[1])
  # Get the frac layer distance of two layer
  frac_layer_dis = layer_dis / len_z
  end_z = frac_layer_dis + start_z
  # # Calculate the rotation angle: theta
  theta = np.arccos(0.5 * (m*m + n*n + 4*m*n) / (m*m + n*n + m*n))
  #-- # Generate the rotation matrix R_frac
  #-- #  R_frac = P.M_cart.P^-1
  #-- inv_primitive_vecs = np.linalg.inv(primitive_vecs)
  #-- rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
  #--                             [np.sin(theta),  np.cos(theta)]])
  #-- rotation_matrix = np.dot(primitive_vecs, rotation_matrix)
  #-- rotation_matrix = np.dot(rotation_matrix, inv_primitive_vecs)
  # Read in the two layers individally
  lay1_super_mult_vec1 = np.array([m, n])
  lay1_super_mult_vec2 = np.array([-n, m+n])
  supercell_number_list, lay1_supercell_vecs, lay1_atom_coord_list = \
    gen_supercell(lay1_super_mult_vec1, lay1_super_mult_vec2, primitive_vecs,
                  atom_coord_list, number_list)
  lay2_super_mult_vec1 = np.array([n, m])
  lay2_super_mult_vec2 = np.array([-m, n+m])
  supercell_number_list, lay2_supercell_vecs, lay2_atom_coord_list = \
    gen_supercell(lay2_super_mult_vec1, lay2_super_mult_vec2, primitive_vecs,
                  atom_coord_list, number_list, mis_frac_x, mis_frac_y)
  supercell_vecs = lay1_supercell_vecs
  # Update the coords list
  tbg_atom_coord_list = []
  for coord_i, _ in enumerate(lay1_atom_coord_list):
    lay1_coord = lay1_atom_coord_list[coord_i]
    lay2_coord = lay2_atom_coord_list[coord_i]
    # The 1st layer
    tbg_coord = np.array([lay1_coord[0], lay1_coord[1], start_z])
    tbg_atom_coord_list.append(tbg_coord)
    # The 2nd layer
    #-- rotate_coord = np.dot(lay2_coord, rotation_matrix)
    tbg_coord = np.array([lay2_coord[0], lay2_coord[1], end_z])
    tbg_atom_coord_list.append(tbg_coord)
  # Recheck if all of the coord are in the cell
  for coord_i, coord in enumerate(tbg_atom_coord_list):
    if (coord[0] > 1.0) or (math.isclose(coord[0], 1.0, rel_tol=FLOAT_PREC)):
      tbg_atom_coord_list[coord_i][0] -= 1
    if coord[0] < 0.0:
      tbg_atom_coord_list[coord_i][0] += 1
    if (coord[1] > 1.0) or (math.isclose(coord[1], 1.0, rel_tol=FLOAT_PREC)):
      tbg_atom_coord_list[coord_i][1] -= 1
    if coord[1] < 0.0:
      tbg_atom_coord_list[coord_i][1] += 1
    if not is_frac_coord_in_cell(tbg_atom_coord_list[coord_i]):
      print("[error] Vector error: [%f, %f, %f]" 
            %(coord[0], coord[1], coord[2]))
  # Update the number list of atoms
  tbg_number_list = [2*val for val in supercell_number_list]
  # RETURN 
  return theta, supercell_vecs, tbg_number_list, tbg_atom_coord_list


def write_TBG_poscar(supercell_vecs, symbols_list, tbg_number_list,
                     tbg_atom_coord_list, comline_args, theta):
  """Write the TBG POSCAR"""
  # Read in necessary parameters
  len_z = comline_args["len_z"]
  m = comline_args["m"]
  n = comline_args["n"]
  layer_dis = comline_args["layer_dis"]
  out_poscar = comline_args["out_poscar"]
  inplance_mis = comline_args["inplane_mis"]
  # Gen the filename of the output POSCAR file
  if out_poscar == "POSCAR.DEFAULT":
    out_poscar = 'POSCAR.%d-%d.vasp' %(m,n)
  ### Gen POSCAR
  out_poscar_lines = []
  # 1st comment line
  out_poscar_lines.append(
    "python TBG.py -m %d -n %d -z %f -d %f -p %s # theta=%f" 
    %(m,n,len_z, layer_dis, inplance_mis, theta))
  out_poscar_lines.append("    1.0")
  # Cell vectors
  for vec in supercell_vecs:
    out_poscar_lines.append("    % 4.8f    % 4.8f    % 4.8f" 
                            %(vec[0], vec[1], 0))
  out_poscar_lines.append("    % 4.8f    % 4.8f    % 4.8f" %(0, 0, len_z))
  # Atoms symbol list
  symbols_list_str = '  '.join(symbols_list)
  out_poscar_lines.append("    %s" %(symbols_list_str))
  # Atoms number list
  tbg_number_list = [str(val) for val in tbg_number_list]
  tbg_number_list_str = '  '.join(tbg_number_list)
  out_poscar_lines.append("    %s" %(tbg_number_list_str))
  # Direct line
  out_poscar_lines.append("Direct")
  # Atoms coords
  for coord in tbg_atom_coord_list:
    out_poscar_lines.append("    % .8f    % .8f    % .8f" 
                            %(coord[0], coord[1], coord[2]))
  # Add \n in each end of line
  for line_i, _ in enumerate(out_poscar_lines):
    out_poscar_lines[line_i] += '\n'
  ### Write the POSCAR to file
  with open(out_poscar, 'w') as fwp:
    fwp.writelines(out_poscar_lines)
  # RETURN
  return 0

  
def main():
  """Main functions"""
  # Get the command line input
  comline_args = get_command_line_input()
  # Read the initial POSCAR info
  primitive_vecs, symbols_list, number_list, atom_coord_list = \
    read_init_poscar(comline_args["init_poscar"])
  # Generate the TBG atoms' coordinates
  theta, supercell_vecs, tbg_number_list, tbg_atom_coord_list = \
    gen_TBG_coords(atom_coord_list, number_list, primitive_vecs, comline_args)
  # Write the TBG POSCAR
  write_TBG_poscar(supercell_vecs, symbols_list, tbg_number_list,
                   tbg_atom_coord_list, comline_args, theta)
  

if __name__ == '__main__':
  main()