import scipy
import numpy as np
import pytools as pt
import reduction
from utils import wrap, resize
from scipy.interpolate import RegularGridInterpolator

RE = 6371000

def get_var(file_name, boxre, var_name, grid_flag):    
    
    f = pt.vlsvfile.VlasiatorReader(file_name)
    size = np.float32(f.get_fsgrid_mesh_size())
    nx = int(size[0])
    nz = int(size[2])

    extents = np.float32(f.get_fsgrid_mesh_extent())
    cellsize = (extents[3] - extents[0]) / nx

    xlim_left = boxre[0] * RE
    xlim_right = boxre[1] * RE
    zlim_bottom = boxre[2] * RE
    zlim_top = boxre[3] * RE
    
    x_ind_left = int(abs((extents[0] - xlim_left) / cellsize))
    x_ind_right = int(abs((extents[0] - xlim_right) / cellsize))
    z_ind_bottom = int(abs((extents[2] - zlim_bottom) / cellsize))
    z_ind_top = int(abs((extents[2] - zlim_top) / cellsize))

    if grid_flag == 'fg':
        var = f.read_fsgrid_variable(var_name)            
        var = var[x_ind_left:x_ind_right,z_ind_bottom:z_ind_top,:]
        var = np.swapaxes(var,0,1)

    elif grid_flag == 'vg':        
        cellids = f.read_variable("CellID")
        var = np.float32(f.read_variable(var_name))    
        
        if var.ndim == 1:
            var=var[cellids.argsort()].reshape(nz,nx)
            var = var[z_ind_bottom:z_ind_top,x_ind_left:x_ind_right]
        elif var.ndim == 2:            
            var=var[cellids.argsort()].reshape(nz,nx,3)
            var = var[z_ind_bottom:z_ind_top, x_ind_left:x_ind_right, :]

    return var


def get_x_points(x_points_file, boxre):
    with open(x_points_file) as f:
        lines = f.readlines()

    x_xp_array = []
    z_xp_array = []

    for k in range(len(lines)):
        lineK = lines[k]
        splitted = lineK.split(' ')
        x_xp = np.divide(float(splitted[0]), RE)
        z_xp = np.divide(float(splitted[2]), RE)
        x_xp_array.append(x_xp)  
        z_xp_array.append(z_xp)  

    # select x points within the domain
    labeling_x = []
    labeling_z = []
    for kkk in range(0, np.array(x_xp_array).shape[0]):
        if x_xp_array[kkk] > boxre[0] and x_xp_array[kkk] < boxre[1] and z_xp_array[kkk] > boxre[2] and z_xp_array[kkk] < boxre[3]:
            labeling_x.append(x_xp_array[kkk])
            labeling_z.append(z_xp_array[kkk])
    return labeling_x, labeling_z


def label_reconnection(labeling_x, labeling_z, B, boxre):
    # create 1D arrays of x and z coordinates
    [xmin, xmax, zmin, zmax] = boxre
    xx = np.linspace(xmin, xmax, (np.array(B).shape[1]))
    zz = np.linspace(zmin, zmax, (np.array(B).shape[0]))

    # label reconnection
    # create zero matrix with size equal to spatial domain size
    reconnection = np.zeros_like(B[:, :, 0])
    for k in range(0, np.array(labeling_x).shape[0]):  # cycle over X-points
        # column index of X-point pixel
        idx_x = (np.abs(xx - np.array(labeling_x)[k])).argmin()
        # row index of X-point pixel
        idx_z = (np.abs(zz - np.array(labeling_z)[k])).argmin()
        # change 0 to 1 for the pixels with X-point
        reconnection[idx_z, idx_x] = 1
    reconnection = wrap(reconnection)
    return reconnection


def get_agyrotropy(file_name, boxre, name_dict):

    B_name = name_dict['B']    
    Pd_name = name_dict['Pd']    
    Pod_name = name_dict['Pod']    

    if B_name == 'B':
        B = get_var(file_name, boxre, var_name=B_name, grid_flag='vg')
    else:
        B = get_var(file_name, boxre, var_name=B_name, grid_flag='fg')
    Pdiag = get_var(file_name, boxre, var_name=Pd_name, grid_flag='vg')
    P0diag = get_var(file_name, boxre, var_name=Pod_name, grid_flag='vg')
    
    print('Pdiag shape', np.shape(Pdiag))
    print('B shape', np.shape(B))

    sx, sy = B.shape[1], B.shape[0]

    Pdiag_flat = Pdiag.reshape(sx * sy, 3)
    P0diag_flat = P0diag.reshape(sx * sy, 3)
    B_flat = B.reshape(sx * sy, 3)

    PTensor = reduction.FullTensor([Pdiag_flat, P0diag_flat])
    PTensor_rotated = reduction.RotatedTensor([PTensor, B_flat])

    aGyrotropy = reduction.aGyrotropy([PTensor_rotated])
    aGyrotropy = aGyrotropy.reshape(sy, sx)

    return aGyrotropy


def get_anisotropy(file_name, boxre, name_dict):

    B_name = name_dict['B']    
    Pd_name = name_dict['Pd']    
    Pod_name = name_dict['Pod']    

    if B_name == 'B':
        B = get_var(file_name, boxre, var_name=B_name, grid_flag='vg')
    else:
        B = get_var(file_name, boxre, var_name=B_name, grid_flag='fg')
    Pdiag = get_var(file_name, boxre, var_name=Pd_name, grid_flag='vg')
    P0diag = get_var(file_name, boxre, var_name=Pod_name, grid_flag='vg')

    sx, sy = B.shape[1], B.shape[0]

    Pdiag_flat = Pdiag.reshape(sx * sy, 3)
    P0diag_flat = P0diag.reshape(sx * sy, 3)
    B_flat = B.reshape(sx * sy, 3)

    PTensor = reduction.FullTensor([Pdiag_flat, P0diag_flat])
    PTensor_rotated = reduction.RotatedTensor([PTensor, B_flat])

    Ppar = reduction.ParallelTensorComponent([PTensor_rotated])
    Pperp = reduction.PerpendicularTensorComponent([PTensor_rotated])
    anisotropy = np.divide(abs(Ppar - Pperp), (Pperp + Ppar))

    anisotropy = anisotropy.reshape(sy, sx)
    return anisotropy


def get_pressure_old(file_name, boxre, name_dict):
    Pd_name = name_dict['Pd']    
    Pdiag = get_var(file_name, boxre, var_name=Pd_name, grid_flag='vg')
    pressure_isotropic = (Pdiag[:, :, 0] + Pdiag[:, :, 1] + Pdiag[:, :, 2]) / 3
    return pressure_isotropic

def get_temperature_old(file_name, boxre, name_dict):         
    rho_name = name_dict['rho']
    pressure = get_pressure_old(file_name, boxre, name_dict)
    rho = get_var(file_name, boxre, rho_name, grid_flag='vg')
    temperature = pressure / rho
    return temperature


def get_temperature(file_name, boxre, name_dict):    
    P_name = name_dict['pressure']    
    rho_name = name_dict['rho']
    pressure = get_var(file_name, boxre, P_name, grid_flag='vg')
    rho = get_var(file_name, boxre, rho_name, grid_flag='vg')
    temperature = pressure / rho
    return temperature


def intp_data(boxre, width, height, nx, ny, data2d):

    x = np.linspace(boxre[0], boxre[1], nx)  
    y = np.linspace(boxre[2], boxre[3], ny)  
    
    xi = np.linspace(boxre[0], boxre[1], width) 
    yi = np.linspace(boxre[2], boxre[3], height) 
    xxi, yyi = np.meshgrid(xi, yi)

    points = np.column_stack(( xxi.ravel(),yyi.ravel() ))
    
    # Create interpolator
    interp_func = RegularGridInterpolator((x,y), data2d.T)
    intp_data = interp_func(points)
    intp_data2d = intp_data.reshape(width,height)

    return intp_data2d

