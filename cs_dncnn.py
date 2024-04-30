import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dip import psnr_mse
from tqdm import tqdm
from dncnn import load_model
from cs_bfdncnn import measurement_matrix
import json
plt.rcParams.update({
    'figure.dpi': 500
})


if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(1580)

    denoiser_name = 'dncnn'
    PnP_iter = 200
    n = 64 * 64
    m_list = [int(n / 4), int(n / 2), int(3 * n / 4)]
    mu = 0.1  # gradient update step size
    sigma = 100  # highest observed noise level in training of denoiser
    A_mode = 'gaussian'  # 'gaussian' or 'dct'

    img = np.array(Image.open('test_image.tiff')) / 255.0
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc + 64, y_loc:y_loc + 64]
    x = patch.reshape(n, 1)

    best_psnrs = []
    best_mses = []
    best_reconstructions = []

    dncnn_denoiser = load_model(f'pretrained/dncnn/bias/0-{int(sigma)}.pt')
    dncnn_denoiser.eval()

    for j, m in enumerate(m_list):
        A = measurement_matrix(A_mode, m, n)
        y = A @ x
        x_hat = np.zeros((n, 1))
        psnr_best, mse_best, x_hat_best = 0., 0., None
        for p in (pbar := tqdm(range(PnP_iter))):
            s = np.clip(x_hat + mu * (A.T @ (y - (A @ x_hat))), 0, 1)
            x_hat = dncnn_denoiser(torch.Tensor(s.reshape(64, 64))[None, None, :, :]).squeeze().detach().numpy().reshape(n, 1)
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
    plt.suptitle(f'DnCNN Projection - pretrained over $\sigma\in$[0, {int(sigma)}] - {A_mode.upper()} A')
    plt.savefig(f'{denoiser_name}_{A_mode}_{int(sigma)}.png')
    plt.show()
