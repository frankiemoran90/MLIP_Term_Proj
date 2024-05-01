import json
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    'figure.dpi': 500
})


def read_json_file(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data


dncnn_gaussian_10 = read_json_file('slides_v1/dncnn_gaussian_10.json')
dncnn_gaussian_100 = read_json_file('slides_v1/dncnn_gaussian_100.json')
bfdncnn_gaussian_10 = read_json_file('slides_v1/bfdncnn_gaussian_10.json')
bfdncnn_gaussian_100 = read_json_file('slides_v1/bfdncnn_gaussian_100.json')
bm3d_gaussian_adaptive = read_json_file('slides_v1/bm3d_gaussian_2.json')
# bm3d_gaussian_nonadaptive = read_json_file('slides_v1/bm3d_gaussian_50.json')
deepdecoder_gaussian = read_json_file('deepdecoder_gaussian.json')

plt.figure(figsize=(6, 4), tight_layout=True)
plt.plot(np.array(dncnn_gaussian_10['m_list'])/4096, dncnn_gaussian_10['best_psnrs'], linestyle="--", marker="o", label="dncnn_10")
plt.plot(np.array(dncnn_gaussian_100['m_list'])/4096, dncnn_gaussian_100['best_psnrs'], linestyle="--", marker="o", label="dncnn_100")
plt.plot(np.array(bfdncnn_gaussian_10['m_list'])/4096, bfdncnn_gaussian_10['best_psnrs'], linestyle="--", marker="o", label="bfdncnn_10")
plt.plot(np.array(bfdncnn_gaussian_100['m_list'])/4096, bfdncnn_gaussian_100['best_psnrs'], linestyle="--", marker="o", label="bfdncnn_100")
# plt.plot(np.array(bm3d_gaussian_nonadaptive['m_list'])/4096, bm3d_gaussian_nonadaptive['best_psnrs'], linestyle="--", marker="o", label="bm3d_gaussian_nonadaptive")
plt.plot(np.array(bm3d_gaussian_adaptive['m_list'])/4096, bm3d_gaussian_adaptive['best_psnrs'], linestyle="--", marker="o", label="bm3d_adaptive")
plt.plot(np.array(deepdecoder_gaussian['m_list'])/4096, deepdecoder_gaussian['best_psnrs'], linestyle="--", marker="o", label="deepdecoder")

plt.xlabel(r'$\frac{m}{n}$  [Fraction of Measurements]')
plt.ylabel('PSNR  [dB]')
plt.title('Gaussian Measurement Matrix A')
plt.xticks([0.25, 0.5, .75])
plt.grid()
plt.legend()
plt.savefig('performance_plot_gaussian.png')
plt.show()
