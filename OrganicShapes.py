import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)
from tqdm import tqdm
def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2) - 0.5 * np.ones((n, 2))
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

def gen_matrix(w, n_shapes=(0, 10), scale=None, max_height=1, rad=0.1, edgy=0.1, npts=(4,6), sample_rate=10):
    mat = np.zeros((w, w))
    scale = (w/5, 0.8*w) if scale is None else scale
    if type(n_shapes) is tuple:
        n_shapes = np.random.randint(n_shapes[0], n_shapes[1])
    polys = []
    heights = []
    # print(n_shapes)

    if n_shapes == 0:
        return mat
    for _ in range(n_shapes):
        scl = np.random.uniform(scale[0], scale[1]) if type(scale) is tuple else scale
        hel = np.random.normal(0.5, 0.25)
        npl = np.random.randint(npts[0], npts[1]) if type(npts) is tuple else npts
        heights.append(hel)

        a = get_random_points(n=npl, scale=scl)
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        xmid = np.random.uniform(0, w)
        ymid = np.random.uniform(0, w)
        x += (xmid - np.average(x))
        y += (ymid - np.average(y))

        lsx = x[-1]
        lsy = y[-1]
        x = x[::10]
        y = y[::10]
        x[-1] = lsx
        y[-1] = lsy
        polygon = Polygon(zip(x, y))
        polys.append(polygon)

    for i in tqdm(range(w), disable=True):
        for j in range(w):
            for h, p in zip(heights, polys):
                if p.contains(Point(i, j)):
                    mat[i, j] += h


    mat *= max_height / np.amax(mat)
    return mat

if __name__ == '__main__':

    mat = gen_matrix(256, n_shapes=(0, 10), scale=None, max_height=1, rad=0.1, edgy=0.1, npts=(4,6), sample_rate=10)
    plt.imshow(mat)
    plt.show()
    assert 1 == 3

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.1
    edgy = 0.1
    polygons = []
    for c in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
        a = get_random_points(n=4, scale=2) + np.random.uniform(-1, 1, 2)
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        lsx = x[-1]
        lsy = y[-1]
        x = x[::10]
        y = y[::10]
        x[-1] = lsx
        y[-1] = lsy
        plt.plot(x, y)
        plt.show()
        polygon = Polygon(zip(x, y))
        polygons.append(polygon)

    r = 256
    xs = np.linspace(-2, 2, r)
    ys = np.linspace(-2, 2, r)
    mat = np.zeros((r, r))
    for i in tqdm(range(r)):
        for j in range(r):
            point = Point(xs[i], ys[j])
            for polygon in polygons:
                if polygon.contains(point):
                    mat[i, j] += 1
    plt.imshow(mat)
    plt.show()



    plt.show()