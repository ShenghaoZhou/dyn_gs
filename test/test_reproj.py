import numpy as np
import pycolmap
import pycolmap.cost_functions
import pyceres

q_wxyz = np.array([0.70710678, 0.0, 0.0, 0.70710678]) # w, x, y, z
q_xyzw = np.array([0.0, 0.0, 0.70710678, 0.70710678]) # x, y, z, w

t = np.array([0., 0., 0.])
P_w = np.array([1.0, 0.0, 5.0])
cam_params = np.array([100.0, 100.0, 50.0, 50.0])
uv = np.array([50.0, 70.0])

cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv)

# Pyceres 2.1+ cost evaluation hack
# In pybind11 PyCeres, CostFunction.__call__ allows evaluating residuals from a list of numpy arrays!
print("w,x,y,z cost:", cost([q_wxyz, t, P_w, cam_params]))
print("x,y,z,w cost:", cost([q_xyzw, t, P_w, cam_params]))
