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


def test_tps():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.optim as optim
    import torch.nn.functional as F

    def to_numpy_image(image: torch.Tensor):
        return (image.detach().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    src = torch.zeros(2, 1, 40, 40)
    src[0, ..., 10:21, 10:21] = 1
    src[1, ..., 28:35, 28:35] = 1

    target = torch.zeros_like(src)
    target[0, ..., 5:-5, 5:-5] = 1
    target[1, ..., 10:-10, 10:-10] = 1

    c_dst = uniform_grid((2, 2)).view(-1, 2)
    theta = torch.zeros(src.shape[0], (c_dst.shape[0] + 2), 2, requires_grad=True)
    size = src.shape
    opt = optim.Adam([theta], lr=1e-2)

    for i in range(400):
        opt.zero_grad()

        grid = tps_grid(theta, c_dst, size)
        warped = F.grid_sample(src, grid)

        loss = F.mse_loss(warped, target)
        loss.backward()
        opt.step()

        if i % 20 == 0:
            print(i, loss.item())

    src_np = to_numpy_image(src).squeeze()
    dst_np = to_numpy_image(target).squeeze()
    final_np = to_numpy_image(warped).squeeze()

    fig, axs = plt.subplots(src.shape[0], 3, figsize=(12, 6))
    for i in range(src.shape[0]):
        axs[i, 0].imshow(src_np[i])
        axs[i, 1].imshow(dst_np[i])
        axs[i, 2].imshow(final_np[i])
        axs[i, 0].set_title('source')
        axs[i, 1].set_title('target')
        axs[i, 2].set_title('result')
    fig.show()


if __name__ == '__main__':
    test_tps()
