import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from thinplate import *


def test_tps():

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
