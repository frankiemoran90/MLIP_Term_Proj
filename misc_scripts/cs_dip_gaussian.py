import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dip import project_DIP, psnr
from tqdm import tqdm


if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(1580)

    img = np.array(Image.open('test_image.tiff')) / 255.0

    PnP_iter = 50
    n = 64 * 64
    m_list = [int(n / 4), int(n / 2), int(3 * n / 4)]
    # m_list = [int(2 * n / 4),]
    mu = 1E-2

    best_psnrs = []
    best_reconstructions = []
    d_4 = [128, 64, 64, 10]

    # x_loc, y_loc = np.random.randint(0, 192, size=2)
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc+64, y_loc:y_loc+64]
    x = patch.reshape(n, 1)

    for j, m in enumerate(m_list):
        S = np.random.normal(0, 1, (m, n))
        A = S / np.linalg.norm(S, axis=0)
        y = A @ x
        x_hat = np.zeros((n, 1))
        # x_hat = np.random.uniform(0, 1, (n, 1))
        # x_hat = A.T @ y
        psnr_best = 0.
        iter_best = 0
        x_hat_best = None
        for p in (pbar := tqdm(range(PnP_iter))):
            s = np.clip(x_hat + mu * (A.T @ (y - (A @ x_hat))), 0, 1)
            x_hat = project_DIP(s.reshape(64, 64), d_4).reshape(n, 1)
            psnr_temp = psnr(x, x_hat, 1)
            if psnr_temp > psnr_best:
                psnr_best = psnr_temp
                x_hat_best = x_hat
                iter_best = p
            pbar.set_postfix({'psnr': psnr_best, 'iter_best': iter_best})
        best_psnrs.append(psnr_best)
        best_reconstructions.append(x_hat_best)

    figs, axes = plt.subplots(1, 1 + len(m_list), figsize=(4 + 4 * len(m_list), 4), tight_layout=True)
    axes = axes.ravel()
    axes[0].imshow(x.reshape(64, 64), cmap='gray')
    axes[0].title.set_text('Original Patch')
    for i, (m, psnr, x_hat) in enumerate(zip(m_list, best_psnrs, best_reconstructions)):
        axes[i+1].imshow(x_hat.reshape(64, 64), cmap='gray')
        axes[i+1].title.set_text(f'$m/n={m/n}$, psnr={psnr:.3f}')
    plt.suptitle('DIP Projection - untrained ')
    plt.savefig('cs_dip_gaussian.png')
    plt.show()
