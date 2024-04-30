# Not working on Apple Silicon, use Colab instead

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import json
from bm3d import bm3d
plt.rcParams.update({
    'figure.dpi': 500
})


def psnr_mse(x, x_hat, x_max):
    n1, n2 = x.shape
    mse = np.linalg.norm(x - x_hat)**2 / (n1 * n2)
    return 10 * np.log10(x_max**2 / mse), mse


def measurement_matrix(mode, m, n):
    if mode == 'gaussian':
        S = np.random.normal(0, 1, (m, n))
    elif mode == 'dct':
        S = np.array([[np.cos(((np.pi * j) / n) * (i + 1/2) ) for i in range(n)] for j in range(m)])
    else:
        print('mode unknown.')
    A = S / np.linalg.norm(S, axis=0)
    return A


if __name__ == '__main__':
    np.random.seed(10)

    denoiser_name = 'bm3d'
    PnP_iter = 200
    n = 64 * 64
    m_list = [int(n / 4), int(n / 2), int(3 * n / 4)]
    mu = 0.1  # gradient update step size
    sigma = 50  # highest observed noise level in training of denoiser
    A_mode = 'gaussian'  # # 'gaussian' or 'dct'

    img = np.array(Image.open('test_image.tiff')) / 255.0
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc+64, y_loc:y_loc+64]
    x = patch.reshape(n, 1)

    best_psnrs = []
    best_mses = []
    best_reconstructions = []

    for j, m in enumerate(m_list):
        A = measurement_matrix(A_mode, m, n)
        y = A @ x
        x_hat = np.zeros((n, 1))
        psnr_best, mse_best, x_hat_best = 0., 0., None
        for p in (pbar := tqdm(range(PnP_iter))):
            s = x_hat + mu * (A.T @ (y - (A @ x_hat)))
            x_hat = bm3d(s.reshape(64, 64, 1), sigma/255.).reshape(n, 1)
            psnr_temp, mse_temp = psnr_mse(x, x_hat, 1)
            if psnr_temp > psnr_best:
                psnr_best = psnr_temp
                mse_best = mse_temp
                x_hat_best = x_hat
                iter_best = p
            pbar.set_postfix({'m/n':m/n, 'psnr_best': psnr_best, 'mse_best':mse_best, 'iter_best': iter_best})
        best_psnrs.append(psnr_best)
        best_mses.append(mse_best)
        best_reconstructions.append(x_hat_best)

    json_log = {
        'denoiser': denoiser_name,
        'PnP_iter': PnP_iter,
        'n': n,
        'm_list': m_list,
        'mu': mu,
        'sigma': sigma,
        'A_mode': A_mode,
        'best_psnrs': best_psnrs,
        'best_mses': best_mses,
    }
    with open(f'{denoiser_name}_{A_mode}_{sigma}.json', 'w', encoding='utf-8') as f:
        json.dump(json_log, f, ensure_ascii=False, indent=4)

    figs, axes = plt.subplots(1, 1 + len(m_list), figsize=(4 + 4 * len(m_list), 4), tight_layout=True)
    axes = axes.ravel()
    axes[0].imshow(x.reshape(64, 64), cmap='gray')
    axes[0].title.set_text('Original Patch')
    for i, (m, psnr, x_hat) in enumerate(zip(m_list, best_psnrs, best_reconstructions)):
        axes[i+1].imshow(x_hat.reshape(64, 64), cmap='gray')
        axes[i+1].title.set_text(f'$m/n={m/n}$, psnr={psnr:.3f}')
    plt.suptitle(f'Bias-Free DnCNN Projection - pretrained over $\sigma\in$[0, {int(sigma)}] - {A_mode.upper()} A')
    plt.savefig(f'{denoiser_name}_{A_mode}_{int(sigma)}.png')
    plt.show()
