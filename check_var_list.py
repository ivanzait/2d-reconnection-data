import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir = '/proj/ivanzait/analysator/'
sys.path.append(module_dir)
import pytools as pt

t=900
filepath = '/wrk-vakka/group/spacephysics/vlasiator/2D/BIB/'
filename = 'bulk.'+str(t).zfill(7)+'.vlsv'
vslv_object=pt.vlsvfile.VlsvReader(filepath+filename)
vslv_object.list()
