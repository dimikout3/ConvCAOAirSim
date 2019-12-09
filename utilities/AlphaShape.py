from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
import shapely.geometry as geometry
from descartes import PolygonPatch
import pylab as pl
import matplotlib.pyplot as plt

# https://gist.github.com/dwyerk/10561690
# https://stackoverflow.com/questions/20474549/extract-points-coordinates-from-a-polygon-in-shapely

def alphashape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """

    coords = points
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    # edge1 = filtered[:,(0,1)]
    # edge2 = filtered[:,(1,2)]
    # edge3 = filtered[:,(2,0)]
    # edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    # m = geometry.MultiLineString(edge_points)
    # triangles = list(polygonize(m))
    # return cascaded_union(triangles), edge_points
    # return cascaded_union(triangles)
    return filtered


def pointInAlphaShape(polygon, detectionPoints, point, alpha=0.9):

    new_points = np.concatenate((detectionPoints, [point]), axis=0)
    t2 = alphashape(new_points,alpha)

    t1 = polygon
    area1 = abs(t1[:,0,0]*(t1[:,1,1]-t1[:,2,1]) + t1[:,1,0]*(t1[:,2,1]-t1[:,0,1]) + t1[:,2,0]*(t1[:,0,1]-t1[:,1,1]))/2

    area2 = abs(t2[:,0,0]*(t2[:,1,1]-t2[:,2,1]) + t2[:,1,0]*(t2[:,2,1]-t2[:,0,1]) + t2[:,2,0]*(t2[:,0,1]-t2[:,1,1]))/2

    areaSum1 = np.sum(area1)
    areaSum2 = np.sum(area2)

    if (areaSum1 == areaSum2):
        return True
    else:
        return False

    # try:
    #     if (polygon == h2).all():
    #         return True
    #     else:
    #         return False
    # except:
    #     """"This means that the polygon and the new hull have different triangles
    #     therefore """"
    #     return False

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    plt.show()
    return fig
