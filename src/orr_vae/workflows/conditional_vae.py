import os
import sys
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import freeze_support

from orr_vae.tool import make_data_loaders_from_json

# ------------------------------
# Argument parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional VAE Training')
    parser.add_argument('--iter', type=int, default=1, 
                       help='Iteration number (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--max_epoch', type=int, default=200,
                       help='Maximum epochs (default: 200)')
    parser.add_argument('--latent_size', type=int, default=32,
                       help='Latent space dimension (default: 32)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta for KL loss (default: 1.0)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Training data ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--load_epoch', type=int, default=-1,
                       help='Load epoch (-1 for no loading, default: -1)')
    parser.add_argument('--base_data_path', type=str,
                       default=str(Path(__file__).parent / "data"),
                       help='Base data directory path')
    parser.add_argument('--result_base_path', type=str,
                       default=str(Path(__file__).parent / "result"),
                       help='Base result directory path')
    parser.add_argument('--label_threshold', type=float, default=0.3,
                       help='Label threshold for binary classification (default: 0.3)')
    parser.add_argument('--grid_x', type=int, default=4,
                       help='Grid size along x (default: 4)')
    parser.add_argument('--grid_y', type=int, default=4,
                       help='Grid size along y (default: 4)')
    parser.add_argument('--grid_z', type=int, default=4,
                       help='Grid size along z / number of slab layers (default: 4)')
    return parser.parse_args()

# ------------------------------
# Hyper-parameter setup
# ------------------------------
# Only parse arguments when executed as a script
if __name__ == "__main__":
    args = parse_args()
else:
    # Provide defaults when imported from another module
    class DefaultArgs:
        iter = 1
        batch_size = 16
        learning_rate = 2e-4
        max_epoch = 200
        latent_size = 32
        beta = 1.0
        train_ratio = 0.9
        seed = 0
        load_epoch = -1
        base_data_path = str(Path(__file__).parent / "data")
        result_base_path = str(Path(__file__).parent / "result")
        label_threshold = 0.3
        grid_x = 4
        grid_y = 4
        grid_z = 4
    args = DefaultArgs()

# Iteration index (global constant to keep compatibility with legacy code)
ITER = args.iter

# Hyper-parameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MAX_EPOCH = args.max_epoch
NUM_WORKERS = 0
LOAD_EPOCH = args.load_epoch

# Dataset configuration
GRID_SIZE = [args.grid_x, args.grid_y, args.grid_z]
TRAIN_RATIO = args.train_ratio
SEED = args.seed

# Latent dimensionality
LATENT_SIZE = args.latent_size
BETA = args.beta

# Derived paths
BASE_DATA_PATH = args.base_data_path
RESULT_BASE_PATH = args.result_base_path
LABEL_THRESHOLD = args.label_threshold

STRUCTURES_DB_PATHS = [
    os.path.join(BASE_DATA_PATH, f"iter{i}_structures.json") for i in range(ITER + 1)
]

OVERPOTENTIALS_JSON_PATHS = [
    os.path.join(BASE_DATA_PATH, f"iter{i}_calculation_result.json") for i in range(ITER + 1)
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device in use: {device}")

# ------------------------------
# Reproducibility
# ------------------------------

def set_seed(seed):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Additional deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ------------------------------
# Conditional VAE definition
# ------------------------------
class ConditionalVAE(nn.Module):
    def __init__(self, latent_size=16, condition_dim=2, structure_layers=4):
        super(ConditionalVAE, self).__init__()
        self.latent_size = latent_size
        self.condition_dim = condition_dim  # overpotential and alloy formation energy
        self.structure_layers = structure_layers
        self.activation = nn.LeakyReLU(0.1, inplace=False)

        # ======== Encoder ========
        # Non-linear transformation of the conditional labels (encoder)
        self.enc_label_fc1 = nn.Linear(condition_dim, 32)
        self.enc_label_fc2 = nn.Linear(32, 32)
        self.enc_label_fc3 = nn.Linear(32, 16)
        
        # Convolutional stack (input channels: layers + 16)
        self.conv1 = nn.Conv2d(self.structure_layers + 16, 128, kernel_size=3, stride=1, padding=1)
        self.enc_gn1 = nn.GroupNorm(16, 128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_gn2 = nn.GroupNorm(16, 256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_gn3 = nn.GroupNorm(32, 512)
        
        # Fully connected mapping to mean and log-variance
        self.fc1 = nn.Linear(512, 128)
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_logvar = nn.Linear(128, latent_size)

        # ======== Decoder ========
        # Non-linear transformation of the conditional labels (decoder)
        self.dec_label_fc1 = nn.Linear(condition_dim, 32)
        self.dec_label_fc2 = nn.Linear(32, 32)
        self.dec_label_fc3 = nn.Linear(32, 16)

        # Fully connected layers (input: latent_size + 16)
        self.dec_fc1 = nn.Linear(latent_size + 16, 64)
        self.dec_fc2 = nn.Linear(64, 128)
        self.dec_fc3 = nn.Linear(128, 64 * 2 * 2)
        
        # Transposed convolutional stack
        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_gn4 = nn.GroupNorm(16, 128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_gn5 = nn.GroupNorm(8, 64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_gn6 = nn.GroupNorm(8, 32)
        self.deconv4 = nn.ConvTranspose2d(32, self.structure_layers * 3, kernel_size=3, stride=1, padding=1)

    def encode_condition_enc(self, y):
        """Conditional embedding for the encoder."""
        h1 = self.activation(self.enc_label_fc1(y))
        h2 = self.activation(self.enc_label_fc2(h1))
        h3 = self.activation(self.enc_label_fc3(h2))
        return h3

    def encode_condition_dec(self, y):
        """Conditional embedding for the decoder."""
        h1 = self.activation(self.dec_label_fc1(y))
        h2 = self.activation(self.dec_label_fc2(h1))
        h3 = self.activation(self.dec_label_fc3(h2))
        return h3

    def encode(self, x, y):
        """Encoder pass that returns the latent mean and log-variance."""
        batch_size = x.size(0)
        
        y_encoded = self.encode_condition_enc(y)  # [B, 16]
        y_expanded = y_encoded.view(batch_size, -1, 1, 1).expand(-1, -1, 8, 8)  # [B, 16, 8, 8]
        x_cond = torch.cat([x, y_expanded], dim=1)  # [B, 20, 8, 8]
        
        h = self.conv1(x_cond)  # [B, 128, 8, 8]
        h = self.enc_gn1(h)
        h = self.activation(h)
        
        h = self.conv2(h)  # [B, 256, 4, 4]
        h = self.enc_gn2(h)
        h = self.activation(h)
        
        h = self.conv3(h)  # [B, 512, 2, 2]
        h = self.enc_gn3(h)
        h = self.activation(h)
        
        # Global Average Pooling
        h = F.adaptive_avg_pool2d(h, (1, 1)).view(batch_size, -1)  # [B, 512]
        
        h = self.fc1(h)  # [B, 128]
        h = self.activation(h)
        
        mu = self.fc_mu(h)      # [B, latent_size]
        logvar = self.fc_logvar(h)  # [B, latent_size]
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """Decoder pass that reconstructs the structure conditioned on ``y``."""
        y_encoded = self.encode_condition_dec(y)  # [B, 16]
        
        z_cat = torch.cat((z, y_encoded), dim=1)  # [B, latent_size + 16]
        
        h = self.dec_fc1(z_cat)  # [B, 64]
        h = self.activation(h)
        
        h = self.dec_fc2(h)  # [B, 128]
        h = self.activation(h)
        
        h = self.dec_fc3(h)  # [B, 256]
        h = self.activation(h)
        
        h = h.view(h.size(0), 64, 2, 2)  # [B, 64, 2, 2]
        
        h = self.deconv1(h)  # [B, 128, 4, 4]
        h = self.dec_gn4(h)
        h = self.activation(h)

        h = self.deconv2(h)  # [B, 64, 8, 8]
        h = self.dec_gn5(h)
        h = self.activation(h)

        h = self.deconv3(h)  # [B, 32, 8, 8]
        h = self.dec_gn6(h)
        h = self.activation(h)

        output = self.deconv4(h)  # [B, 12, 8, 8]
        
        return output

    def forward(self, x, y):
        """Forward propagation."""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

# ------------------------------
# Loss function
# ------------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1):
    """
    Scaled VAE loss made of reconstruction and KL divergence terms.

    recon_x: [B, 3*L, 8, 8] - decoder logits (L = number of layers)
    x: [B, L, 8, 8] - integer class labels (0, 1, 2)
    """
    x = x.to(dtype=torch.long)
    
    class_weights = torch.tensor([0.1, 1.0, 1.0], device=x.device)
    
    n_layers = x.shape[1]
    expected_channels = n_layers * 3
    if recon_x.shape[1] != expected_channels:
        raise ValueError(
            f"Decoder output channel mismatch: got {recon_x.shape[1]}, expected {expected_channels}"
        )

    recon_loss = 0
    for z in range(n_layers):
        layer_pred = recon_x[:, z*3:(z+1)*3]  # [B, 3, 8, 8]
        layer_target = x[:, z]  # [B, 8, 8]
        
        recon_loss += F.cross_entropy(
            layer_pred, 
            layer_target, 
            weight=class_weights,
            reduction='sum' 
        )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ------------------------------
# Training and evaluation helpers
# ------------------------------
def train_vae(epoch, model, train_loader, optimizer, beta):
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    start_time = time.time()
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        try:
            data = data.to(device).float()
            labels = labels.to(device).float()  # [B, 2]
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data, labels)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")
                
        except Exception as e:
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl, elapsed

def test_vae(epoch, model, test_loader, beta):
    model.eval()
    test_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, labels in test_loader:
            try:
                data = data.to(device).float()
                labels = labels.to(device).float()  # [B, 2]
                
                recon, mu, logvar = model(data, labels)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
                
                test_loss += loss.item()
                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                
            except Exception as e:
                traceback.print_exc()
                continue
    
    elapsed = time.time() - start_time
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = recon_loss_total / len(test_loader.dataset)
    avg_kl = kl_loss_total / len(test_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl, elapsed

# ------------------------------
# Plotting helpers
# ------------------------------
def plot_learning_curves(train_losses, test_losses, result_dir):
    """Save loss curves as individual figures."""
    epochs = np.arange(1, len(train_losses) + 1)
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    # 1. Total Loss
    plt.figure(figsize=(12, 12))
    plt.plot(epochs, train_losses[:, 0], 'b-', label='Train Total Loss', linewidth=3)
    plt.plot(epochs, test_losses[:, 0], 'r-', label='Test Total Loss', linewidth=3)
    plt.title('Total Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/total_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Reconstruction Loss
    plt.figure(figsize=(12, 12))
    plt.plot(epochs, train_losses[:, 1], 'b-', label='Train Recon Loss', linewidth=3)
    plt.plot(epochs, test_losses[:, 1], 'r-', label='Test Recon Loss', linewidth=3)
    plt.title('Reconstruction Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/reconstruction_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. KL Loss
    plt.figure(figsize=(12, 12))
    plt.plot(epochs, train_losses[:, 2], 'b-', label='Train KL Loss', linewidth=3)
    plt.plot(epochs, test_losses[:, 2], 'r-', label='Test KL Loss', linewidth=3)
    plt.title('KL Divergence Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/kl_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved learning curves:")
    print(f"  - {result_dir}/total_loss.png")
    print(f"  - {result_dir}/reconstruction_loss.png")
    print(f"  - {result_dir}/kl_loss.png")

# ------------------------------
# Main routine
# ------------------------------
def main():
    print("=== Conditional VAE configuration ===")
    print(f"iter: {ITER}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"max_epoch: {MAX_EPOCH}")
    print(f"latent_size: {LATENT_SIZE}")
    print(f"beta: {BETA}")
    print(f"train_ratio: {TRAIN_RATIO}")
    print(f"seed: {SEED}")
    print(f"load_epoch: {LOAD_EPOCH}")
    print(f"base_data_path: {BASE_DATA_PATH}")
    print(f"result_base_path: {RESULT_BASE_PATH}")
    print(f"label_threshold: {LABEL_THRESHOLD}")
    print(f"grid_size: {GRID_SIZE}")
    print("=" * 40)
    
    result_dir = os.path.join(RESULT_BASE_PATH, f"iter{ITER}")
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"=== Conditional VAE training (iter{ITER}) ===")
    print(f"Using datasets: iter0 to iter{ITER}")
    print(f"Structure DB files: {len(STRUCTURES_DB_PATHS)}")
    print(f"Overpotential JSON files: {len(OVERPOTENTIALS_JSON_PATHS)}")
    print(f"Result directory: {result_dir}")
    
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch = MAX_EPOCH
    num_workers = NUM_WORKERS
    load_epoch = LOAD_EPOCH
    latent_size = LATENT_SIZE
    beta = BETA

    set_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader, test_loader, dataset = make_data_loaders_from_json(
        structures_db_paths=STRUCTURES_DB_PATHS,
        overpotentials_json_paths=OVERPOTENTIALS_JSON_PATHS,
        use_binary_labels=True,
        train_ratio=TRAIN_RATIO,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=SEED,
        grid_size=GRID_SIZE,
        label_threshold=LABEL_THRESHOLD
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    model = ConditionalVAE(
        latent_size=latent_size,
        condition_dim=2,
        structure_layers=GRID_SIZE[2],
    ).to(device)
    print("Conditional VAE initialised (two conditional labels: overpotential and alloy formation energy)")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(load_epoch + 1, max_epoch):
        print(f"\nEpoch {epoch + 1}/{max_epoch} started")
        
        train_loss, train_recon, train_kl, train_time = train_vae(
            epoch, model, train_loader, optimizer, beta
        )
        test_loss, test_recon, test_kl, test_time = test_vae(
            epoch, model, test_loader, beta
        )
        
        print(f"Epoch: {epoch+1}/{max_epoch}")
        print(f"Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Total: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
        print(f"Train time: {train_time:.2f}s, Eval time: {test_time:.2f}s")
        
        train_loss_list.append([train_loss, train_recon, train_kl])
        test_loss_list.append([test_loss, test_recon, test_kl])
    
    train_loss_array = np.array(train_loss_list)
    test_loss_array = np.array(test_loss_list)
    
    np.save(f"{result_dir}/train_loss.npy", train_loss_array)
    np.save(f"{result_dir}/test_loss.npy", test_loss_array)
    
    plot_learning_curves(train_loss_array, test_loss_array, result_dir)
    
    torch.save(model.state_dict(), f"{result_dir}/final_cvae_iter{ITER}.pt")
    
    print(f"Training artefacts saved to {result_dir}")
    print(f"Stored model: final_cvae_iter{ITER}.pt")

if __name__ == "__main__":
    freeze_support()
    main()
