import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from glob import glob
from skimage import io
from thop import profile
from tqdm.auto import tqdm
from skimage.transform import resize
from torch.utils.data import DataLoader
from model.mfennet_model import MFEnNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


volumes_path = 'path/to/train/imgs'
labels_path = 'path/to/train/labels'

# get sorted list of image and mask files
volume_files = sorted(glob(os.path.join(volumes_path, '*')))
label_files = sorted(glob(os.path.join(labels_path, '*')))


# read images into numpy arrays
img_size_in = 256
img_size_out = img_size_in

volumes = []
for f in tqdm(volume_files):
  volumes.append(resize(io.imread(f), (img_size_in, img_size_in), preserve_range=True, anti_aliasing=True).astype(np.uint8))
volumes = np.array(volumes)

labels = []
for f in tqdm(label_files):
  labels.append(resize(io.imread(f), (img_size_out, img_size_out), preserve_range=True, anti_aliasing=True).astype(np.uint8))
labels = np.array(labels)


# convert to torch and normalize
# volumes shape is (N, H, W, C) - need to convert to (N, C, H, W)
volumes = torch.Tensor(volumes).permute(0, 3, 1, 2)/255  # Convert from (N, H, W, C) to (N, C, H, W)  # for rgb image
# volumes = torch.Tensor(volumes)[:, None, :, :]/255  # Add channel dimension for labels    # for gray image
labels = torch.Tensor(labels)[:, None, :, :]/255  # Add channel dimension for labels

print("volumes shape:", volumes.shape)
print("labels shape:", labels.shape)


# construct dataset
dataset = torch.utils.data.TensorDataset(volumes, labels)

# set dataloadaer
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# instantiate model
network = MFEnNet(3, 1)
network = network.to(device)

# define criterion and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(network.parameters(), lr=2e-4)

# count model parameters and FLOPs
input_size=(1, 3, img_size_in, img_size_in)
dummy_input = torch.randn(*input_size).to(device)
flops, params = profile(network, inputs=(dummy_input,), verbose=False)
print(f"Number of parameters: {params:,}")
print(f"Number of FLOPs (Multiply-Adds): {flops:,}")
print(f"Number of FLOPs (GFLOPs): {flops/1e9:.2f} GFLOPs")


# train
n_epochs = 50

model_name = 'mfennet'
dataset_name = 'casia'

saved_mode_name = f'{model_name}_model_{dataset_name}_in{img_size_in}_{n_epochs}e'
log_file_name = f'loss_log_{saved_mode_name}.txt'

for epoch in range(n_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for mini_images, mini_labels in tqdm(dataloader):
        mini_images = mini_images.to(device)
        mini_labels = mini_labels.to(device)
        preds = network(mini_images)
        loss = criterion(preds, mini_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"\nEpoch: {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

    # save model loss after every epoch
    if not os.path.exists(log_file_name):
        with open(log_file_name, "w") as f:
            pass
    with open(log_file_name, "a") as f:
        f.write(f"{epoch+1},{avg_loss}\n")


# save model
torch.save(network.state_dict(), f'{saved_mode_name}.pth')
print(f"Model saved!")