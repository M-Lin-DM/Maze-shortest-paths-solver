import numpy as np
from numba import njit
import torch


@njit
def eikonal_update_cpu(T, y, x, Lx, F=None):
    """Update for iterative solution of the eikonal equation on CPU."""

    minx = np.minimum(T[y * Lx + x - 1], T[y * Lx + x + 1])  # element wise min
    miny = np.minimum(T[(y - 1) * Lx + x], T[(y + 1) * Lx + x], )
    mina = np.minimum(T[(y - 1) * Lx + x - 1], T[(y + 1) * Lx + x + 1])
    minb = np.minimum(T[(y - 1) * Lx + x + 1], T[(y + 1) * Lx + x - 1])

    if F is None:
        A = np.where(np.abs(mina - minb) >= 2, np.minimum(mina, minb) + np.sqrt(2),
                     (1. / 2) * (mina + minb + np.sqrt(4 - (mina - minb) ** 2)))
        B = np.where(np.abs(miny - minx) >= np.sqrt(2), np.minimum(miny, minx) + 1,
                     (1. / 2) * (miny + minx + np.sqrt(2 - (miny - minx) ** 2)))

        return np.sqrt(A * B)

    elif F is not None:
        A = np.where(np.abs(mina - minb) >= 2 / F[y * Lx + x], np.minimum(mina, minb) + np.sqrt(2) / F[y * Lx + x],
                     (1. / 2) * (mina + minb + np.sqrt(4 * (1 / F[y * Lx + x]) ** 2 - (mina - minb) ** 2)))
        B = np.where(np.abs(miny - minx) >= np.sqrt(2) / F[y * Lx + x], np.minimum(miny, minx) + 1 / F[y * Lx + x],
                     (1. / 2) * (miny + minx + np.sqrt(2 * (1 / F[y * Lx + x]) ** 2 - (miny - minx) ** 2)))

        return np.sqrt(A * B)


def eikonal_update_gpu(T, y, x, Lx, F=None):
    """Update for iterative solution of the eikonal equation on GPU. Lower pixel values tend to propagate outwards, as if the floor is collapsing downward."""

    minx = torch.minimum(T[y * Lx + x - 1], T[y * Lx + x + 1])  # element wise min
    miny = torch.minimum(T[(y - 1) * Lx + x], T[(y + 1) * Lx + x], )
    mina = torch.minimum(T[(y - 1) * Lx + x - 1], T[(y + 1) * Lx + x + 1])
    minb = torch.minimum(T[(y - 1) * Lx + x + 1], T[(y + 1) * Lx + x - 1])

    if F is None:
        A = torch.where(torch.abs(mina - minb) >= 2, torch.minimum(mina, minb) + np.sqrt(2),
                        (1. / 2) * (mina + minb + torch.sqrt(4 - (mina - minb) ** 2)))
        B = torch.where(torch.abs(miny - minx) >= np.sqrt(2), torch.minimum(miny, minx) + 1,
                        (1. / 2) * (miny + minx + torch.sqrt(2 - (miny - minx) ** 2)))

        return torch.sqrt(A * B)

    elif F is not None:
        A = torch.where(torch.abs(mina - minb) >= 2 / F[y * Lx + x],
                        torch.minimum(mina, minb) + np.sqrt(2) / F[y * Lx + x],
                        (1. / 2) * (mina + minb + torch.sqrt(4 * (1 / F[y * Lx + x]) ** 2 - (mina - minb) ** 2)))
        B = torch.where(torch.abs(miny - minx) >= np.sqrt(2) / F[y * Lx + x],
                        torch.minimum(miny, minx) + 1 / F[y * Lx + x],
                        (1. / 2) * (miny + minx + torch.sqrt(2 * (1 / F[y * Lx + x]) ** 2 - (miny - minx) ** 2)))

        return torch.sqrt(A * B)

def extend_centers(T, y, x, Lx, niter, equation='eikonal', F=None, use_gpu=False):
    """ For niter iterations, update the distance field T on CPU

    Parameters
    --------------
    T: float32, array
        _ x Lx array that diffusion is run in
    y: int32, array
        y coords of pixels inside mask
    x: int32, array
        x coords of pixels inside mask
    Lx: int32
        size of x-dimension of mask
    niter: int32
        number of iterations to run diffusion
    equation: str
        equation to use for updating the distance field. Choose either "eikonal" or "heat"
    F: numpy ndarray
        Speed field used in Eikonal equation. If left as None, there is uniform speed everywhere, equivalent to setting
        F=1 everywhere. Must be same shape as loaded image. Eikonal eq: F = 1/magnitude(gradient(T))

    Returns
    ---------------
    T/Tg: float64, numpy ndarray or pytorch tensor if use_gpu=True
        amount of diffused particles at each pixel
    """
    if use_gpu is False:
        for t in range(niter):
            if t % 50 == 0:
                print(f"{t} of {niter}")
            if equation is "eikonal":
                T[y * Lx + x] = eikonal_update_cpu(T, y, x, Lx, F)  # solve eikonal equation
            elif equation is "heat":
                # solve heat equation
                T[y * Lx + x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                                          T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                                          T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                                          T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])

        return T

    elif use_gpu is True:
        device = torch.device('cuda')

        Tg = torch.from_numpy(T).to(device)  # T is provided to this func as float32 np.array not float64
        yg = torch.from_numpy(y).to(device)
        xg = torch.from_numpy(x).to(device)
        Lxg = torch.from_numpy(Lx).to(device)

        if F is not None:
            F = torch.from_numpy(F).to(device)

        # print("Tg.device: ", Tg.device)
        # see if computation on gpu works
        # try:
        #     d = yg + xg
        #     print(d)
        # except Exception as e:
        #     print(e)

        for t in range(niter):
            if t % 50 == 0:
                print(f"{t} of {niter}")
            if equation is "eikonal":
                Tg[yg * Lxg + xg] = eikonal_update_gpu(Tg, yg, xg, Lxg, F)  # solve eikonal equation
            elif equation is "heat":
                # solve heat equation
                Tg[yg * Lxg + xg] = 1 / 9. * (Tg[yg * Lxg + xg] + Tg[(yg - 1) * Lxg + xg] + Tg[(yg + 1) * Lxg + xg] +
                                              Tg[yg * Lxg + xg - 1] + Tg[yg * Lxg + xg + 1] +
                                              Tg[(yg - 1) * Lxg + xg - 1] + Tg[(yg - 1) * Lxg + xg + 1] +
                                              Tg[(yg + 1) * Lxg + xg - 1] + Tg[(yg + 1) * Lxg + xg + 1])

        return Tg
