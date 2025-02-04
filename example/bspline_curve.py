from GCT.curve_generator import curve_generator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    point1 = np.array([ [1], [5]])
    point2 = np.array([ [5], [3]])
    point3 = np.array([ [6], [5]])
    point4 = np.array([ [2], [2]])

    point_list = [point1, point2, point3, point4]

    cg = curve_generator()

    curve = cg.generate_curve('bspline', point_list, 0.1, 2)

    if curve is not None: cg.plot_curve(curve, show_way_points=True , show_direction=False)

    plt.show()