import numpy as np
from scipy import interpolate

def generate_bspline_path(way_points=[], step_size = 0.1, **kwargs):

    """
    Arguments:
        way_points: the way points of the curve: 3* 1 matrix
        step_size: the distance between each point

        Fit cubic clamped uniform spline interpolants to way points
    """

    # calc Chord length
    dim = len(way_points[0])
    pts = np.array(way_points).flatten().reshape(len(way_points),dim).transpose()
    diff_sqr = np.diff(pts[0:2, :])**2
    arc_len = np.sqrt(diff_sqr[0,:] + diff_sqr[1,:])
    arc_cum = np.concatenate(([0], arc_len.cumsum()))

    # fitting cubic spline using chord length
    tck,u=interpolate.splprep([pts[0,:], pts[1,:]])
    sample_num = max(100, int(arc_cum[-1]/step_size))   # at least 100 pts
    x_i, y_i = interpolate.splev(np.linspace(0, 1, sample_num), tck)

    path_list = []
    for i in range(len(x_i)):
        next_pose = np.zeros([dim,1]);
        next_pose[0,0] = x_i[i]
        next_pose[1,0] = y_i[i]
        path_list.append(next_pose)

    if len(path_list) != 0:
        if np.linalg.norm(way_points[-1] - path_list[-1]) <= 0.01:
            path_list[-1] = way_points[-1]
        else:
            path_list.append(way_points[-1])
    else:
        path_list.append(way_points[-1])
    return path_list