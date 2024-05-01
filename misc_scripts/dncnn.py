import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.serialization import default_restore_location
import argparse


def psnr(x, x_hat, x_max):
    n1, n2 = x.shape
    mse = np.linalg.norm(x - x_hat)**2 / (n1 * n2)
    return 10 * np.log10(x_max**2 / mse)


class BFBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias = False, affine=True):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)

        self.use_bias = use_bias;

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0,1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / ( self.running_var.view(-1, 1)**0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mu
                    self.running_var = (1-self.momentum)*self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1,1)
            y = y / (sigma2.view(-1,1)**.5 + self.eps)

        if self.affine:
            y = self.weight.view(-1, 1) * y;
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0,1)


class DnCNN(nn.Module):
    """DnCNN as defined in https://arxiv.org/abs/1608.03981
       reference implementation: https://github.com/SaoYan/DnCNN-PyTorch"""

    def __init__(self, depth=20, n_channels=64, image_channels=1, bias=False, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1

        self.bias = bias;
        if not bias:
            norm_layer = BFBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.depth = depth;

        self.first_layer = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size,
                                     padding=padding, bias=self.bias)

        self.hidden_layer_list = [None] * (self.depth - 2);

        self.bn_layer_list = [None] * (self.depth - 2);

        for i in range(self.depth - 2):
            self.hidden_layer_list[i] = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                                  kernel_size=kernel_size, padding=padding, bias=self.bias);
            self.bn_layer_list[i] = norm_layer(n_channels)

        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);
        self.bn_layer_list = nn.ModuleList(self.bn_layer_list);
        self.last_layer = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size,
                                    padding=padding, bias=self.bias)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=1, help="number of channels")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
        parser.add_argument("--bias", action='store_true', help="use residual bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels=args.in_channels, n_channels=args.hidden_size, depth=args.num_layers, bias=args.bias)

    def forward(self, x):
        y = x
        out = self.first_layer(x);
        out = F.relu(out);

        for i in range(self.depth - 2):
            out = self.hidden_layer_list[i](out);
            out = self.bn_layer_list[i](out);
            out = F.relu(out)

        out = self.last_layer(out);

        return y - out


def load_model(checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
    args = argparse.Namespace(**{ **vars(state_dict["args"]), "no_log": True})

    model = DnCNN.build_model(args).to(device)
    model.load_state_dict(state_dict["model"][0])
    model.eval()
    return model


if __name__ == '__main__':
    np.random.seed(1580)
    torch.manual_seed(1580)

    device = 'cpu'
    sigma = 50

    img = np.array(Image.open('test_image.tiff'))
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc + 64, y_loc:y_loc + 64]
    noise = np.random.normal(0, sigma, size=patch.shape)

    patch_noisy = patch + noise
    dncnn_denoiser = load_model('pretrained/dncnn/bias/0-10.pt')
    patch_hat = dncnn_denoiser(torch.Tensor(patch_noisy/255.0)[None, None, :, :])

    figs, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    axes = axes.ravel()
    axes[0].imshow(patch, cmap='gray')
    axes[0].title.set_text('Original Patch')
    axes[1].imshow(patch_noisy, cmap='gray')
    axes[1].title.set_text(f'psnr={psnr(patch, patch_noisy, 255):.3f}')
    axes[2].imshow(patch_hat.squeeze().detach().numpy(), cmap='gray')
    axes[2].title.set_text(f'psnr={psnr(patch, patch_hat.squeeze().detach().numpy() * 255, 255):.3f}')
    plt.show()
