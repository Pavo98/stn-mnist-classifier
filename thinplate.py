# Based on work:
#   [1] http://www.zemris.fer.hr/~ssegvic/project/pubs/palasek12ms.pdf.
#   [2] https://github.com/cheind/py-thin-plate-spline
#
import torch


def tps(theta, ctrl, grid, eps=1e-6):
    r"""Evaluate the thin-plate-spline (TPS) at x,y locations defined in grid

        TPS(x, y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_i=0,T theta[i] U(x, y, ctrl[i])  # from [1] (2.12)

    where U is radial basis function - polyharmonic spline with beta = 2

    Params
    ------
    :param theta: Nx(T+3)x2 tensor
        N - Batch size
        T+3 model parameters for T control points in dx, dy
    :param ctrl: NxTx2 tensor or Tx2 tensor
        T control points in normalized coordinates range [0, 1]
    :param grid: NxHxWx3 tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate
    :param eps: scalar, Optional (default: 1e-6)
        Zero value threshold for log function

    Returns
    -------
    :return z: NxHxWx2 tensor
        Function values at each grid location in dx and dy.
    """
    N, H, W, _ = grid.size()

    if ctrl.dim() == 2:
        ctrl = ctrl.expand(N, *ctrl.size())

    T = ctrl.shape[1]

    diff = grid[..., 1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
    D = torch.sqrt((diff ** 2).sum(-1))
    U = (D ** 2) * torch.log(D + eps)  # shape: NxHxWxT

    w, a = theta[:, :-3, :], theta[:, -3:, :]

    reduced = theta.shape[1] == T + 2
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1)

    # shapes: U: Nx(HxW)xT, w: NxTx2 ==> b: NxHxWx2
    b = torch.bmm(U.view(N, -1, T), w).view(N, H, W, 2)
    # shapes: grid: Nx(HxW)x3, a: Nx3x2 ==> z: NxHxWx2
    z = torch.bmm(grid.view(N, -1, 3), a).view(N, H, W, 2) + b

    return z


def tps_grid(theta, ctrl, size):
    """Compute thin-plate-spline grid for sampling

    Params
    ------
    :param theta: Nx(T+3)x2 tensor
        N - Batch size
        T+3 - parameters for T control points in dx, dy
    :param ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized coordinates range [0, 1]
    :param size: tuple
        Output grid size as NxCxHxW.

    Returns
    -------
    :return grid: NxHxWx2 tensor
        Grid for image sampling.
    """
    N, C, H, W = size

    grid = theta.new_empty(N, H, W, 3)
    grid[..., 0] = 1.
    grid[..., 1] = torch.linspace(0, 1, W)
    grid[..., 2] = torch.linspace(0, 1, H).unsqueeze(-1)

    z = tps(theta, ctrl, grid)
    return (grid[..., 1:] + z) * 2 - 1


def uniform_grid(shape):
    """Uniform control points in grid across normalized image coordinates

    Params
    ------
    :param shape: tuple
        HxW number of control points

    Returns
    -------
    :return points: HxWx2 tensor
        Control points over [0, 1] normalized image range.
    """
    H, W = shape[:2]
    c = torch.zeros(H, W, 2)
    c[..., 0] = torch.linspace(0, 1, W)
    c[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1)
    return c
