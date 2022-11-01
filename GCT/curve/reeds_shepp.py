"""
Code Author: Han Ruihua
Reference: paper 'Optimal paths for a car that goes both forwards and backwards', github: https://github.com/nathanlct/reeds-shepp-curves/tree/master

timeflip: 
  positive to negative; 
  LpSpLp --> LnSnLn
  x, y, phi --> -x, y, -phi  

reflect:
  left to right; 
  LpSpLp -- > RpSpRp
  x, y, phi --> x, -y, -phi 

backward: 
    reverse the operation
    LpRnLn --> LnRnLp  
    x, y, phi --> xcos(phi) + ysin(phi), sin(phi) - ycos(phi), phi 
"""

import numpy as np
from collections import namedtuple
from math import sqrt, atan2, sin, cos, pi, inf, asin, acos


element = namedtuple('element','len steer gear') # steer 1,0,-1, gear 1,-1

def generate_reeds_shepp(start=np.zeros((3, 1)), end=np.ones((3, 1)), min_radius=1, step_size=0.1, include_gear=False):

    """
    Arguments:
        start: the start point of the curve: 3* 1 matrix
        end: the end point of the curve: 3 * 1 matrix
        min_radius: The minimum turning radius of the car
        step_size: the distance between each point
    """

    if isinstance(start_point, tuple):
        start_point = np.array([start_point]).T

    if isinstance(goal_point, tuple):
        goal_point = np.array([goal_point]).T

    x, y, phi = preprocess(start, end, min_radius)

    path_formula1 = [LpSpLp, LpSpRp, LpRnLp, LpRpLnnRn, LpRnLnRp, LpRnRnLnRp]
    path_formula2 = [LpRnLn, LpRnSnLn, LpRnSnRn]

    path_list1, List1 = symmetry_curve1(x, y, phi, path_formula1)
    path_list2, List2 = symmetry_curve2(x, y, phi, path_formula2)

    total_path_list = path_list1 + path_list2
    total_L_list = List1 + List2
    
    L_min = min(total_L_list) 
    path_min = total_path_list[total_L_list.index(L_min)]

    path_point_list = path_generate(start_point, path_min, min_radius, step_size, include_gear)

    path_point_list.append(goal_point)

def preprocess(start, end, min_radius):
    # preprocess: get coordinates of end in the set of axis where start is (0,0,0) with unit distance

    ro_phi = start[2, 0]
    ro_matrix = np.array([[cos(ro_phi), sin(ro_phi)], 
                        [-sin(ro_phi), cos(ro_phi)]])

    diff = end[0:2] - start[0:2]
    new_pos = ro_matrix @ diff

    new_x = new_pos[0, 0] / min_radius
    new_y = new_pos[1, 0] / min_radius
    new_phi = M(end[2, 0] - start[2, 0])

    return new_x, new_y, new_phi

# calculate curves
def symmetry_curve1(x, y, phi, path_formula1):
    
    path_list = []
    L_list = []

    for timeflip in [False, True]:
        for reflect in [False, True]:
            for cal_formula in path_formula1:

                rg_flag = -1 if timeflip else 1
                rs_flag = -1 if reflect else 1
                
                path, L = cal_formula(rg_flag*x, rs_flag*y, rs_flag*rg_flag*phi, timeflip=timeflip, reflect=reflect)

                path_list.append(path.copy())
                L_list.append(L)
    
    return path_list, L_list

def symmetry_curve2(x, y, phi, path_formula2):
    
    path_list = []
    L_list = []

    for timeflip in [False, True]:
        for reflect in [False, True]:
            for backward in [False, True]:
                for cal_formula in path_formula2:

                    rg_flag = -1 if timeflip else 1 
                    rs_flag = -1 if reflect else 1
                    sx, sy, sphi = reverse(x,y,phi) if backward else (x,y,phi)
                    
                    path, L = cal_formula(rg_flag*sx, rs_flag*sy, rs_flag*rg_flag*sphi, timeflip=timeflip, reflect=reflect, backward=backward)

                    path_list.append(path.copy())
                    L_list.append(L)
    
    return path_list, L_list

def path_generate(start_point, path, min_radius, step_size, include_gear=False):
    path_point_list = [start_point]
    end_point = None

    if len(path) == 0:
        print('no path')
        return path_point_list

    for i in range(len(path)):
        
        path_list, end_point = element_sample(path[i], start_point, min_radius, step_size, include_gear)

        path_point_list = path_point_list + path_list
        start_point = end_point

    return path_point_list

def element_sample(element, start_point, min_radius, step_size, include_gear=False):

    length = element.len * min_radius
    
    if include_gear:
        if start_point.shape[0] < 4:
            add_row = np.array([[element.gear]])
            start_point = np.vstack((start_point, add_row))
        else:
            start_point[3, 0] = element.gear

    path_list = []
    cur_length = 0

    while cur_length < length:   
        
        pre_length = cur_length + step_size

        if cur_length <= length and pre_length > length:
            step_size = length - cur_length

        new_point = self.motion_acker_step(start_point, element.gear, element.steer, step_size, include_gear)

        cur_length = cur_length + step_size
        path_list.append(new_point)
        start_point = new_point

    return path_list, new_point

# trans to (-pi, pi)
def M(theta):
    theta = theta % (2*pi)
    if theta < - pi: return theta + 2*pi
    if theta >= pi: return theta - 2*pi
    return theta

# Polar coordinate  
def R(x, y):   
    r = sqrt(x**2 + y**2)
    theta = atan2(y, x)
    return r, theta

# for backwards (reverse)
def reverse(x, y, phi):
    new_x = x*cos(phi) + y*sin(phi)
    new_y = x*sin(phi) - y*cos(phi)
    return new_x, new_y, phi


# curve formula

# formula 8.1
def LpSpLp(x, y, phi, timeflip=False, reflect=False):
    
    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    #calculate
    u, t = R(x - sin(phi), y-1+cos(phi))
    v = (phi - t) % (2*pi)

    if t<0 or u<0 or v<0:
        return path, inf

    path.append(element(t, steer_flag, gear_flag)) 
    path.append(element(u, 0, gear_flag)) 
    path.append(element(v, steer_flag, gear_flag))

    L = abs(t) + abs(u) + abs(v)

    return path, L

# formula 8.2
def LpSpRp(x, y, phi, timeflip=False, reflect=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    u1, t1 = R(x+sin(phi), y-1-cos(phi))

    if u1**2 < 4:
        L = inf
    else:
        u = sqrt(u1**2 - 4)

        T, theta = R(u, 2)
        t = M(t1+theta)
        v = M(t-phi)
        L = abs(t) + abs(u) + abs(v)

        if t<0 or u<0 or v<0:
            return path, inf

        path.append(element(t, steer_flag*1, gear_flag)) 
        path.append(element(u, 0, gear_flag)) 
        path.append(element(v, steer_flag*-1, gear_flag))   

    return path, L

# formula 8.3  typo in paper
def LpRnLp(x, y, phi, timeflip=False, reflect=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x - sin(phi)
    eta = y - 1 + cos(phi)
    u1, theta = R(xi, eta)

    # if u1 ** 2 > 4:
    #     return path, inf
    if u1 > 4:
        return path, inf
    
    A = acos(u1/4)
    t = M(theta + pi/2 + A)
    u = M(pi - 2*A)
    v = M(phi-t-u)
    
    L = abs(t) + abs(u) + abs(v)

    if t<0 or u<0 or v<0:
        return path, inf

    path.append(element(t, steer_flag*1, gear_flag*1)) 
    path.append(element(u, steer_flag*-1, gear_flag*-1)) 
    path.append(element(v, steer_flag*1, gear_flag*1)) 

    return path, L

# formula 8.4, typo in paper, 
def LpRnLn(x, y, phi, timeflip=False, reflect=False, backward=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x - sin(phi)
    eta = y - 1 + cos(phi)
    u1, theta = R(xi, eta)

    if u1  > 4:
        return path, inf
    
    A = acos(u1/4)
    t = M(theta + pi/2 + A)
    u = M(pi - 2*A)
    v = M(t+u-phi)
    
    if t<0 or u<0 or v<0:
        return path, inf
    
    L = abs(t) + abs(u) + abs(v)

    if backward:
        path.append(element(v, steer_flag*1, gear_flag*-1))
        path.append(element(u, steer_flag*-1, gear_flag*-1))
        path.append(element(t, steer_flag*1, gear_flag*1))
    else:
        path.append(element(t, steer_flag*1, gear_flag*1)) 
        path.append(element(u, steer_flag*-1, gear_flag*-1)) 
        path.append(element(v, steer_flag*1, gear_flag*-1)) 

    return path, L

# formula 8.7 typo in paper
def LpRpLnnRn(x, y, phi, timeflip=False, reflect=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x + sin(phi)
    eta = y - 1 - cos(phi)
    u1, theta = R(xi, eta)

    if u1 > 4:
        return path, inf 
    
    if u1 <= 2:
        A = acos((u1+2)/4)
        t = M(theta+pi/2+A)
        u = M(A)
        v = M(phi-t+2*u)
    else:
        A = acos((u1-2)/4)
        t = M(theta+pi/2-A)
        u = M(pi-A)
        v = M(phi-t+2*u)
    
    if t<0 or u<0 or v<0:
        return path, inf

    L = abs(t) + 2*abs(u) + abs(v)

    path.append(element(t, steer_flag*1, gear_flag*1)) 
    path.append(element(u, steer_flag*-1, gear_flag*1)) 
    path.append(element(u, steer_flag*1, gear_flag*-1))
    path.append(element(v, steer_flag*-1, gear_flag*-1))  

    return path, L

# formula 8.8 
def LpRnLnRp(x, y, phi, timeflip=False, reflect=False):
    
    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x + sin(phi)
    eta = y - 1 - cos(phi)
    u1, theta = R(xi, eta)
    rho = (20 - u1**2) / 16

    if rho >= 0 and rho <= 1 and u1<=6:
        u = acos(rho)
        A = asin(2*sin(u)/u1)
        t = M(theta+pi/2+A)
        v = M(t-phi)

        if t<0 or u<0 or v<0:
            return path, inf
        
        L = abs(t) + 2*abs(u) + abs(v)

        path.append(element(t, steer_flag*1, gear_flag*1)) 
        path.append(element(u, steer_flag*-1, gear_flag*-1)) 
        path.append(element(u, steer_flag*1, gear_flag*-1))
        path.append(element(v, steer_flag*-1, gear_flag*1))  

    else:
        return path, inf

    return path, L

# formula 8.9
def LpRnSnLn(x, y, phi, timeflip=False, reflect=False, backward=False):
    
    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x - sin(phi)
    eta = y - 1 + cos(phi)
    rho, theta = R(xi, eta)
    
    if rho < 2:
        return path, inf
    
    u = sqrt(rho**2-4) -2
    A = atan2(2, u+2)
    t = M(theta+pi/2+A)
    v = M(t-phi+pi/2)

    if t<0 or u<0 or v<0:
        return path, inf
    
    L = abs(t) + pi/2 + abs(u) + abs(v)

    if backward:
        path.append(element(v, steer_flag*1, gear_flag*-1)) 
        path.append(element(u, steer_flag*0, gear_flag*-1))
        path.append(element(pi/2, steer_flag*-1, gear_flag*-1)) 
        path.append(element(t, steer_flag*1, gear_flag*1))               
    else:
        path.append(element(t, steer_flag*1, gear_flag*1)) 
        path.append(element(pi/2, steer_flag*-1, gear_flag*-1)) 
        path.append(element(u, steer_flag*0, gear_flag*-1))
        path.append(element(v, steer_flag*1, gear_flag*-1)) 

    return path, L

# formula 8.10
def LpRnSnRn(x, y, phi, timeflip=False, reflect=False, backward=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x + sin(phi)
    eta = y - 1 - cos(phi)
    rho, theta = R(xi, eta)

    if rho < 2:
        return path, inf

    t = M(theta+pi/2)
    u = rho-2
    v = M(phi - t -pi/2)

    if t<0 or u<0 or v<0:
        return path, inf

    L = abs(t) + pi/2 + abs(u) + abs(v)
    
    if backward:
        path.append(element(v, steer_flag*-1, gear_flag*-1))
        path.append(element(u, steer_flag*0, gear_flag*-1)) 
        path.append(element(pi/2, steer_flag*-1, gear_flag*-1)) 
        path.append(element(t, steer_flag*1, gear_flag*1))
    else:
        path.append(element(t, steer_flag*1, gear_flag*1)) 
        path.append(element(pi/2, steer_flag*-1, gear_flag*-1)) 
        path.append(element(u, steer_flag*0, gear_flag*-1))
        path.append(element(v, steer_flag*-1, gear_flag*-1))

    return path, L

# formula 8.11 typo in paper
def LpRnRnLnRp(x, y, phi, timeflip=False, reflect=False):

    path = []
    gear_flag = -1 if timeflip else 1
    steer_flag = -1 if reflect else 1

    xi = x + sin(phi)
    eta = y - 1 - cos(phi)
    rho, theta = R(xi, eta)

    if rho < 4:
        return path, inf

    u = sqrt(rho**2 - 4) - 4
    A = atan2(2, u+4)
    t = M(theta+pi/2+A)
    v = M(t-phi)

    if t<0 or u<0 or v<0:
        return path, inf

    L = abs(t) + pi/2 + abs(u) + pi/2 + abs(v)

    path.append(element(t, steer_flag*1, gear_flag*1)) 
    path.append(element(pi/2, steer_flag*-1, gear_flag*-1)) 
    path.append(element(u, steer_flag*0, gear_flag*-1))
    path.append(element(pi/2, steer_flag*1, gear_flag*-1))
    path.append(element(v, steer_flag*-1, gear_flag*1))

    return path, L