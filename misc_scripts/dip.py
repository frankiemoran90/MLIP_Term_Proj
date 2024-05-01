import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def psnr_mse(x, x_hat, x_max):
    n1, n2 = x.shape
    mse = np.linalg.norm(x - x_hat)**2 / (n1 * n2)
    return 10 * np.log10(x_max**2 / mse), mse


class DIP(nn.Module):
    def __init__(self,depth_vec):
        super(DIP, self).__init__()
        # Layer 1: Conv+ReLU+Batch+Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(depth_vec[0], depth_vec[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_vec[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Layer 2: Conv+ReLU+Batch+Upsample
        self.up2 = nn.Sequential(
            nn.Conv2d(depth_vec[1], depth_vec[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_vec[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        #Layer 3: Conv+ReLU+Batch+Upsample
        self.up3 = nn.Sequential(
            nn.Conv2d(depth_vec[2], depth_vec[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_vec[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # Layer 4: Conv+ReLU+Batch+Upsample
        self.up4 = nn.Sequential(
            nn.Conv2d(depth_vec[3], 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Final Layer: Conv+Sigmoid,
        self.final = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0),  # Maintains spatial dimensions
            nn.Sigmoid()  # Sigmoid activation to bound the output
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)  # Final convolution followed by sigmoid activation
        return x


# Input:
#     x: np \times np numpy array, representing an image with normalized entries between (0,1)
#     depth_vec: a numpy array of length 4 containing [L1,L2,L3,L4] as shown in Fig. 1
# Output:
#     x_hat: np \times np numpy array, representing the projection of x into the output of the DIP described by depth_vec

def project_DIP(x, depth_vec, device='cpu'):
    # Initialize the input with Gaussian noise
    p, p = x.shape
    n_input = int(p / 16)

    input_tensor = torch.randn(1, depth_vec[0], n_input, n_input).to(device)  # (batch_size, channels, height, width)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()   # Convert to tensor if it's a numpy array
    x = x[None, None, :, :].to(device)

    # Initialize the model and optimizer
    model = DIP(depth_vec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1E-4, weight_decay=5E-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000)
    criterion = nn.MSELoss()

    # Training loop
    num_steps = 200  # Adjust the number of steps as needed
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

    x_hat = output.to('cpu').squeeze().detach().numpy()

    return x_hat


if __name__ == '__main__':
    np.random.seed(1580)
    torch.manual_seed(1580)

    depth_vec = [128, 64, 64, 10]
    device = 'cpu'

    img = np.array(Image.open('test_image.tiff')) / 255.0
    x_loc, y_loc = 30, 120  # upper-left corner location
    patch = img[x_loc:x_loc + 64, y_loc:y_loc + 64]

    patch_hat = project_DIP(patch, depth_vec, device)

    figs, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    axes = axes.ravel()
    axes[0].imshow(patch, cmap='gray')
    axes[0].title.set_text('Original Patch')
    axes[1].imshow(patch_hat, cmap='gray')
    axes[1].title.set_text(f'psnr={psnr(patch, patch_hat, 1):.3f} | {depth_vec}')
    plt.suptitle('DIP Image Fit')
    plt.show()
