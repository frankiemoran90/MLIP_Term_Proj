import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

def psnr(x, x_hat, x_max):
    n1, n2 = x.shape
    mse = np.linalg.norm(x - x_hat)**2 / (n1 * n2)
    return 10 * np.log10(x_max**2 / mse)


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def decodernw(
        num_output_channels=3,
        num_channels_up=[128] * 4,
        filter_size_up=1,
        need_sigmoid=True,
        pad='reflection',
        upsample_mode='bilinear',
        act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act=False,
        bn_affine=True,
        upsample_first=True,
):
    num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    model = nn.Sequential()

    for i in range(len(num_channels_up) - 1):

        if upsample_first:
            model.add(conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))
            if upsample_mode != 'none' and i != len(num_channels_up) - 2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode != 'none' and i != 0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
            model.add(conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))

        if i != len(num_channels_up) - 1:
            if (bn_before_act):
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))
            model.add(act_fun)
            if (not bn_before_act):
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))

    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def project_DeepDecoder(x, prev_model=None, device='cpu', gt=None):
    # Initialize the input with Gaussian noise
    p, p = x.shape
    n_input = int(p / 16)

    input_tensor = torch.randn(1, 128, n_input, n_input).to(device)  # (batch_size, channels, height, width)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()   # Convert to tensor if it's a numpy array
    x = x[None, None, :, :].to(device)

    # Initialize the model and optimizer
    if prev_model is None:
        model = decodernw(num_output_channels=1).to(device)
    else:
        model = prev_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1E-3, weight_decay=5E-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[700, 900])
    criterion = nn.MSELoss()

    # Training loop
    num_steps = 200  # Adjust the number of steps as needed
    # for step in (pbar := tqdm(range(num_steps))):
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(input_tensor)

        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # pbar.set_postfix({'psnr': 10 * np.log10(1.**2 / loss.item()),
        #                   'mse': loss.item(),
        #                   'lr': optimizer.param_groups[0]['lr']})
        # print(np.linalg.norm(output.to('cpu').squeeze().detach().numpy() - gt)**2 / (64 * 64))

    x_hat = output.to('cpu').squeeze().detach().numpy()

    return x_hat, model


if __name__ == '__main__':
    np.random.seed(1580)
    torch.manual_seed(1580)

    device = 'cpu'

    img = np.array(Image.open('test_image.tiff')) / 255.0
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc + 64, y_loc:y_loc + 64]

    patch_hat, model = project_DeepDecoder(patch, device=device, gt=patch)

    figs, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    axes = axes.ravel()
    axes[0].imshow(patch, cmap='gray')
    axes[0].title.set_text('Original Patch')
    axes[1].imshow(patch_hat, cmap='gray')
    axes[1].title.set_text(f'psnr={psnr(patch, patch_hat, 1):.3f}')
    plt.suptitle('DeepDecoder Image Fit')
    plt.show()