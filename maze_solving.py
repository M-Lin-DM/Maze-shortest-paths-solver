import matplotlib.pyplot as plt
import numpy as np
from core import extend_centers
from utils import normalize_field
from PIL import Image
import time
from utils import is_row_in_array, rot_m


class EikonalSolver:
    """ Class used for computation of the distance field within an image from a certain point (called end_point here)
    The field comes from solving the Eikonal equation iteratively, in find_distance_field(), extend_centers().

    Parameters
    --------------
    img_path: str
        image path. Image should be a binary png image in which prohibited regions are black and all other regions are white. The optimized path will only be allowed to enter white areas.
    niter: int
        number of iterations fed to run extend_centers(). The number of updates of all pixels in the distance field.
    equation: str
        equation to use for updating the distance field. Choose either "eikonal" or "heat"
    use_gpu: bool
        Whether to use pytorch and move tensors to the gpu for speed. Make sure pytorch, cudatoolkit,
        and cudnn are installed!
    F: numpy ndarray
        Speed field used in Eikonal equation. If left as None, there is uniform speed everywhere, equivalent to setting
        F=1 everywhere. Must be same shape as loaded image. Eikonal eq: F = 1/magnitude(gradient(T))
    """

    def __init__(self, img_path, niter, equation="eikonal", use_gpu=False, F=None):
        self.img_path = img_path
        self.niter = niter
        self.equation = equation
        self.F = F
        self.use_gpu = use_gpu

    def get_masks(self):
        img = Image.open(self.img_path)
        img = np.asarray(img)
        print(img.shape, img.dtype)
        mask = np.round(img / 255).astype(np.float32)  # black interior, white walls
        mask_inverted = np.round(mask * -1 + 1)  # white interior, black walls
        return mask, mask_inverted

    def get_start_and_end_points(self, mask):
        fig1, ax1 = plt.subplots()
        ax1.imshow(mask, cmap='gray')
        ax1.set_title("mask: should have black interior and white walls")

        print(
            "Please click starting point and then ending point. Right click to remove a point. When last point is clicked, all crosses will disappear. Close figure to finalize choice. Distance field will be with respect to the end point clicked.")
        points = plt.ginput(2, show_clicks=True, mouse_add=1, mouse_pop=3)
        plt.show()
        return np.array(list(points[0])).astype(int), np.array(list(points[1])).astype(int)

    def find_distance_field(self):
        margin = 0

        wall_height = self.niter  # note: setting large OR max countour level = niter*0.8 may be useful for visualization of propagating wave since only a fraction of the maze is plotted

        mask, mask_inverted = self.get_masks()
        start_point, end_point = self.get_start_and_end_points(mask)
        print(f"start and end point: {start_point}, {end_point}")
        # set radiation center. Pokes pixel-sized hole in interior region that will act as a seed for the wave. Low values propagate outward, as if a sinkhole is opening up the ground.
        mask_inverted[end_point[1], end_point[0]] = 0

        Lx = mask.shape[1]
        Ly = mask.shape[0]

        # define the coords of pixels that will be updated in the algorithm. Only white pixels will be updated, thus, the black regions (0) always "bleed into" the white (1)
        y, x = np.nonzero(
            mask_inverted)  # returns row, col indices of nonzero elements. Only these positions will be updated by the algorithm.

        if not self.use_gpu:
            y = y.astype(np.int32)
            x = x.astype(np.int32)
            Ly = np.int32(Ly)  # this causes error when using the GPU, but may be necessary for the cpu version
            Lx = np.int32(Lx)
        elif self.use_gpu:
            x = np.array(x)
            y = np.array(y)
            Ly = np.array(Ly)  # this causes error when using the GPU, but may be necessary for the cpu version
            Lx = np.array(Lx)

        y2, x2 = y.copy(), x.copy()
        # restrict updateable pixels to those not on the border of the image. pixels on border have incomplete neighborhoods, causing indexing errors

        x2 = x2[(y > margin) & (y < Ly - (margin + 1)) & (x > margin) & (x < Lx - (margin + 1))]
        y2 = y2[(y > margin) & (y < Ly - (margin + 1)) & (x > margin) & (x < Lx - (margin + 1))]

        # increase value in wall areas to wall_height so wavefront cant encroach. These pixels wont act as seeds, propagating low values.
        T = mask * wall_height
        # increase value of border pixels to make them impassible too.
        T[:, 0] = wall_height
        T[:, Lx - 1] = wall_height
        T[0, :] = wall_height
        T[Ly - 1, :] = wall_height

        T = T.flatten()

        # bottleneck command:
        t0 = time.perf_counter()
        if not self.use_gpu:
            T = extend_centers(T, y2, x2, Lx, self.niter, equation=self.equation, F=self.F, use_gpu=self.use_gpu)
            T = T.reshape((Ly, Lx))
        elif self.use_gpu:
            T = extend_centers(T, y2, x2, Lx, self.niter, equation=self.equation, F=self.F, use_gpu=self.use_gpu)
            T = T.reshape((Ly, Lx)).cpu()  # move T back to cpu.
            T = T.numpy()  # T is still a pytorch tensor. Leaving it as such caused errors with later plotting functions that use numpy
        t1 = time.perf_counter()
        t_elapsed = t1 - t0
        print(f"t_elapsed for extend_centers(): {t_elapsed}")

        dy, dx = np.gradient(
            T)  # output is same size as T. NOTE: the gradient is returned as d(axis 0), d(axis 1) ie dy, dx
        mu = np.stack((-dx, -dy))  # shape (2, Ly, Lx)
        mu = normalize_field(mu)
        varbs = {'T': T, 'mu': mu, 'x': x2, 'y': y2, 'start_point': start_point, 'end_point': end_point,
                 'mask': mask}  #

        return T, varbs


class PathFinder:
    """ Class used for computation of the distance field within an image from a certain point (called end_point here)
    The field comes from solving the Eikonal equation iteratively, in find_distance_field(), extend_centers().

    Parameters
    --------------
    N_steps: int
        Number of iterations to run the find_path() algorithm. The length of the trajectory returned.
    eta: float
        The scale factor appled to distance field gradient vector v in obatining the next point: p_next = p + v * self.eta
    start_point: ndarray[int], shape: (1,2)
        intital pixel location of the path

    _varbs: dict
    x: ndarray[int]
        x coordinates of allowed pixels (the interior, non-wall pixels where the distance field is computed). The path can also only move within this set.
    y: ndarray[int]
        y coordinates of allowed pixels
    mu ndarray[float] shape: (2, Ly, Lx)
        gradient of the distance field
    """

    def __init__(self, _varbs, N_steps=1000, eta=1):
        self.N_steps = N_steps
        self.eta = eta
        self.start_point = _varbs['start_point']
        self.x = _varbs['x']
        self.y = _varbs['y']
        self.mu = _varbs['mu']

    def find_path(self):
        # Starting at start_point, push the path forward for N_steps iterations.
        # At each step, we add a point to the path by moving in the direction of the local gradient held in mu.
        # If the proposed point enters the set of prohibited pixels, we rotate the step vector v until it is
        # back in the set of allowed pixels.

        traj = np.zeros((self.N_steps, 2))
        traj[0] = self.start_point
        p = np.array(self.start_point).astype(float)
        dat = np.concatenate((self.x[:, None], self.y[:, None]), axis=1).astype(int)  # interior coordinates

        for t in range(1, self.N_steps):
            if t % 50 == 0:
                print(f"{t} of {self.N_steps}")
            if not is_row_in_array(p.astype(int), dat):
                print("p outside allowed coords")
                break

            v = self.mu[:, int(p[1]), int(p[0])]  # local gradient vector at rounded position (which should be in dat)
            p_next = p + v * self.eta

            while not is_row_in_array(p_next.astype(int),
                                      dat):  # checks if the rounded next position is inside the allowed coordintates dat
                v2 = rot_m(np.pi / 15) @ v  # rotate step vector by fixed amount
                p_next = p + v2 * self.eta  # redefine p_next using rotated vector

                v = v2.copy()
            if np.all(p == p_next.astype(float)):
                raise Exception("point not moving. Distance field may not have reached start_point. Try increasing the number of iterations given to EikonalSolver.")
            p = p_next.astype(float)
            traj[t] = p

        return traj
