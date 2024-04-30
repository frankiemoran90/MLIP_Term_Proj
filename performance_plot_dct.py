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


dncnn_dct_10 = read_json_file('slides_v1/dncnn_dct_10.json')
dncnn_dct_100 = read_json_file('slides_v1/dncnn_dct_100.json')
bfdncnn_dct_10 = read_json_file('slides_v1/bfdncnn_dct_10.json')
bfdncnn_dct_100 = read_json_file('slides_v1/bfdncnn_dct_100.json')
bm3d_dct_adaptive = read_json_file('slides_v1/bm3d_dct_2.json')
# bm3d_dct_nonadaptive = read_json_file('slides_v1/bm3d_dct_50.json')
deepdecoder_dct = read_json_file('deepdecoder_dct.json')

plt.figure(figsize=(6, 4), tight_layout=True)
plt.plot(np.array(dncnn_dct_10['m_list'])/4096, dncnn_dct_10['best_psnrs'], linestyle="--", marker="o", label="dncnn_10")
plt.plot(np.array(dncnn_dct_100['m_list'])/4096, dncnn_dct_100['best_psnrs'], linestyle="--", marker="o", label="dncnn_100")
plt.plot(np.array(bfdncnn_dct_10['m_list'])/4096, bfdncnn_dct_10['best_psnrs'], linestyle="--", marker="o", label="bfdncnn_10")
plt.plot(np.array(bfdncnn_dct_100['m_list'])/4096, bfdncnn_dct_100['best_psnrs'], linestyle="--", marker="o", label="bfdncnn_100")
# plt.plot(np.array(bm3d_dct_nonadaptive['m_list'])/4096, bm3d_dct_nonadaptive['best_psnrs'], linestyle="--", marker="o", label="bm3d_nonadaptive")
plt.plot(np.array(bm3d_dct_adaptive['m_list'])/4096, bm3d_dct_adaptive['best_psnrs'], linestyle="--", marker="o", label="bm3d_adaptive")
plt.plot(np.array(deepdecoder_dct['m_list'])/4096, deepdecoder_dct['best_psnrs'], linestyle="--", marker="o", label="deepdecoder")

plt.xlabel(r'$\frac{m}{n}$  [Fraction of Measurements]')
plt.ylabel('PSNR  [dB]')
plt.title('DCT Measurement Matrix A')
plt.xticks([0.25, 0.5, .75])
plt.grid()
plt.legend()
plt.savefig('performance_plot_dct.png')
plt.show()
