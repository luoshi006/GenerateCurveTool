import numpy as np
from pyclothoids import Clothoid

def generate_clothoid_path(start=np.zeros((3, 1)), end=np.ones((3, 1)), step_size=0.1, **kwargs):

    """
    Arguments:
        start: the start point of the curve: 3* 1 matrix
        end: the end point of the curve: 3 * 1 matrix
        step_size: the distance between each point
    """

    clothoid0 = Clothoid.G1Hermite(start[0,0], start[1,0], start[2,0], end[0,0], end[1,0], end[2,0])
    # print(clothoid0.dk, clothoid0.KappaStart, clothoid0.KappaEnd)

    real_length = clothoid0.length

    # generate a series of sample points
    cur_length = 0
    path_list = []

    while cur_length <= real_length - step_size:
        cur_length = cur_length + step_size
        next_pose = np.zeros((3,1))
        next_pose[0,0] = clothoid0.X(cur_length)
        next_pose[1,0] = clothoid0.Y(cur_length)
        next_pose[2,0] = clothoid0.Theta(cur_length)
        path_list.append(next_pose)

    if len(path_list) != 0:
        if np.linalg.norm(end - path_list[-1]) <= 0.01:
            path_list[-1] = end
        else:
            path_list.append(end)
    else:
        path_list.append(end)

    return path_list