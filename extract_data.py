import os
import sys
module_dir = '/proj/ivanzait/analysator/'
sys.path.append(module_dir)
import pytools as pt

import argparse
from constants import Re, xmin, xmax, zmin, zmax, runs, height, width
from utils import wrap, resize
#import reduction
import features
import numpy as np

######## MAIN C#########

boxre = [xmin, xmax, zmin, zmax]

############################################################

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-o', '--outdir', type=str)
arg_parser.add_argument('-x', '--x_dir', default='x_points', type=str)
args = arg_parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

for run in runs:

    run_id = run['id']
    start_time = run['t_min']
    end_time = run['t_max']

    if run_id == 'BCH':
        bulk_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BCH/bulk/'
        x_dir = '/wrk-vakka/group/spacephysics/vlasiator/2D/BCH/x_and_o_points/'
        # naming:
        E_name = 'E'
        B_name = 'B'
        rho_name = 'rho'
        V_name = 'rho_v'
        Pd_name = 'PTensorDiagonal'
        Pod_name = 'PTensorOffDiagonal'

    if run_id == 'BGF':
        bulk_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk/'
        x_dir = '/wrk-vakka/group/spacephysics/vlasiator/2D/BGF/ivan/x_and_o_points/'
        # naming:
        E_name = 'fg_e'
        B_name = 'vg_b_vol'
        rho_name = 'proton/vg_rho'
        V_name = 'proton/vg_v'
        Pd_name = 'proton/vg_ptensor_diagonal'
        Pod_name = 'proton/vg_ptensor_offdiagonal'

    if run_id == 'BGD':
        bulk_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BGD/continuation/bulk/'
        x_dir = '/wrk-vakka/group/spacephysics/vlasiator/2D/BGD/x_and_o_points/'
        # naming:
        E_name = 'fg_e'
        B_name = 'vg_b_vol'
        rho_name = 'proton/vg_rho'
        V_name = 'proton/vg_v'
        Pd_name = 'proton/vg_ptensor_diagonal'
        Pod_name = 'proton/vg_ptensor_offdiagonal'


    if run_id == 'BCQ':
        bulk_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BCQ/bulk/'
        x_dir = '/wrk-vakka/group/spacephysics/vlasiator/2D/BCQ/x_and_o_points/'
        # naming:
        E_name = 'E'
        B_name = 'B'
        rho_name = 'rho'
        V_name = 'rho_v'
        Pd_name = 'PTensorDiagonal'
        Pod_name = 'PTensorOffDiagonal'

    if run_id == 'BIB':
        bulk_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BIB/'
        x_dir = '/wrk-vakka/group/spacephysics/vlasiator/2D/BIB/x_and_o_points/'
        # naming:
        E_name = 'fg_e'
        B_name = 'vg_b_vol'
        rho_name = 'proton/vg_rho'
        V_name = 'proton/vg_v'
        Pd_name = 'proton/vg_ptensor_diagonal'
        Pod_name = 'proton/vg_ptensor_offdiagonal'


    name_list = [E_name, B_name, rho_name, V_name, Pd_name, Pod_name]

    for t in range(start_time, end_time + 1):
        print('run', run_id, 't=', str(t))

        # Loading x points
        # x_loc_file = f'{args.x_dir}/{run_id}_x_points_{t}.txt'
        x_loc_file = x_dir + 'x_point_location_' + str(t) + '.txt'
        labeling_x, labeling_z = features.get_x_points(x_loc_file, boxre)
        

        # read simulation data
        file_name = bulk_path + 'bulk.' + str(t).zfill(7) + '.vlsv'        

        B = features.get_var(file_name, boxre, B_name, grid_flag='vg')
        B_mag = np.linalg.norm(B, axis=-1)
        Bx, By, Bz = B[:, :, 0], B[:, :, 1], B[:, :, 2]

        if run_id == 'BCH':
            E = features.get_var(file_name, boxre, E_name, grid_flag='vg')
        elif run_id == 'BCQ':    
            E = features.get_var(file_name, boxre, E_name, grid_flag='vg')
        elif run_id == 'BGF':
            E = features.get_var(file_name, boxre, E_name, grid_flag='fg')
        elif run_id == 'BGD':
            E = features.get_var(file_name, boxre, E_name, grid_flag='fg')

        E_mag = np.linalg.norm(E, axis=-1)
        Ex, Ey, Ez = B[:, :, 0], E[:, :, 1], E[:, :, 2]

        rho = features.get_var(file_name, boxre, rho_name, grid_flag='vg')
        earth_mask = rho == 0

        v = features.get_var(file_name, boxre, V_name, grid_flag='vg')
        v_mag = np.linalg.norm(v, axis=-1)
        vx, vy, vz = v[:, :, 0], v[:, :, 1], v[:, :, 2]

        # calculate isotropic pressure and temperature
        pressure = features.get_pressure(
            file_name=file_name, boxre=boxre, name_list=name_list)
        temperature = features.get_temperature(
            file_name=file_name, boxre=boxre, name_list=name_list)

        # calculate pressure agyrotropy and anisotropy
        anisotropy = features.get_anisotropy(
            file_name=file_name, boxre=boxre, name_list=name_list)
        agyrotropy = features.get_agyrotropy(
            file_name=file_name, boxre=boxre, name_list=name_list)
        reconnection = features.label_reconnection(
            labeling_x, labeling_z, B, boxre)

        # interpolation
        var_list = [
            B_mag,
            Bx,
            By,
            Bz,
            E_mag,
            Ex,
            Ey,
            Ez,
            v_mag,
            vx,
            vy,
            vz,
            rho,
            pressure,
            temperature,
            agyrotropy,
            anisotropy,
            reconnection]

        ny, nx = rho.shape[0], rho.shape[1]
        for var in var_list:
            var = features.intp_data(boxre, width, height, nx, ny, data2d=var)

        # concatenate processed features
        frame_data = np.stack(var_list, axis=-1)
        frame_data[earth_mask] = 0

        np.save(f'{args.outdir}/{run_id}_{t}.npy', resize(frame_data))

        print(f'Extracted frame {run_id}_{t}')
