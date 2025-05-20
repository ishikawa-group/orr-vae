import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# matplotlib設定（日本語フォント警告を避けるため）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

# 自作モジュールのインポート
from catalyst_ccVAE_2 import CVAE

# GPUが利用可能ならそれを使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

result_dir = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE")
model_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE/final_model.pt")

def load_model(model_path, latent_size=64):
    """Load trained model"""
    model = CVAE(latent_size=latent_size, condition_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def analyze_tensor(tensor):
    """Get basic statistical information about tensor (distinguish even/odd layers)"""
    # Class mapping: 0=Empty, 1=Pd, 2=Pt
    # Visualization mapping: 0=Empty → 0, 1=Pd → 1, 2=Pt → 2.0
    
    # ID conversion mapping
    id_mapping = {0: 0.0, 1: 1.0, 2: 2.0}
    
    # Overall statistics
    id0_count = (tensor == 0).sum().item()  # Empty
    id1_count = (tensor == 1).sum().item()  # Pd
    id2_count = (tensor == 2).sum().item()  # Pt
    
    value_counts = {
        0: int(id0_count),  # Empty
        1: int(id1_count),  # Pd
        2: int(id2_count)   # Pt
    }
    
    # Map for visualization
    viz_tensor = torch.zeros_like(tensor, dtype=torch.float32)
    for cls_idx, viz_val in id_mapping.items():
        viz_tensor[tensor == cls_idx] = viz_val
    
    # Calculate statistics for even and odd layers
    even_layers_stats = []  # Even layers (0, 2, 4, ...)
    odd_layers_stats = []   # Odd layers (1, 3, 5, ...)
    
    # Statistics for each layer
    layer_stats = []
    for i in range(tensor.size(0)):
        layer = tensor[i]
        viz_layer = viz_tensor[i]
        
        # Extract different elements for even and odd layers
        if i % 2 == 0:  # Even layer
            # Consider only elements at even indices
            elements = layer[0::2, 0::2]
            viz_elements = viz_layer[0::2, 0::2]
        else:  # Odd layer
            # Consider only elements at odd indices
            elements = layer[1::2, 1::2]
            viz_elements = viz_layer[1::2, 1::2]
        
        # Count for each class
        layer_id0_count = (elements == 0).sum().item()  # Empty
        layer_id1_count = (elements == 1).sum().item()  # Pd
        layer_id2_count = (elements == 2).sum().item()  # Pt
        
        layer_stats.append({
            'mean': viz_elements.mean().item(),
            'std': viz_elements.std().item(),
            'min': viz_elements.min().item(),
            'max': viz_elements.max().item(),
            'median': torch.median(viz_elements).item(),
            # Class distribution counts
            'id_counts': {
                0: int(layer_id0_count),  # Empty
                1: int(layer_id1_count),  # Pd
                2: int(layer_id2_count)   # Pt
            }
        })
        
        # Store statistics for even and odd layers separately
        if i % 2 == 0:
            even_layers_stats.append(layer_stats[-1])
        else:
            odd_layers_stats.append(layer_stats[-1])
    
    return {
        'global_mean': viz_tensor.mean().item(),
        'global_std': viz_tensor.std().item(),
        'value_counts': value_counts,
        'layer_stats': layer_stats,
        'even_layers_stats': even_layers_stats,
        'odd_layers_stats': odd_layers_stats,
        'viz_tensor': viz_tensor  # Tensor converted for visualization
    }

@torch.no_grad()
def test_conditional_outputs(model, result_dir):
    """Test output tensors with different conditions and save results"""
    # Use fixed latent variable (for reproducibility)
    # torch.manual_seed(42)
    
    output_file = result_dir / f"ccVAE_condition_test.txt"
    
    overpotentials = np.arange(0.2, 0.7, 0.1)  # From 0.2 to 0.7 in steps of 0.1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Conditional VAE Model Output Test\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Latent size: {model.latent_size}\n\n")
        f.write("Test conditions:\n")
        f.write(f"- Fixed latent variable: Seed 42\n")
        f.write(f"- Overpotential range: 0.2V to 0.7V\n\n")
        f.write("Class representation:\n")
        f.write(f"- Class 0: Empty (display value: 0)\n")
        f.write(f"- Class 1: Pd (display value: 1)\n")
        f.write(f"- Class 2: Pt (display value: 2)\n\n")
        f.write("Even layers: Elements at even indices (0,2,4,...)\n")
        f.write("Odd layers: Elements at odd indices (1,3,5,...)\n\n")
        f.write("=" * 60 + "\n\n")
        
        all_tensors = []  # Store all output tensors
        all_value_counts = []  # For statistics
        
        for op in overpotentials:
            print(f"Testing output at overpotential {op:.2f}V...")
            
            # Encode overpotential condition
            y = torch.tensor([[op]], dtype=torch.float).to(device)
            label_embedding = model.label_encoder(y)
            
            # Combine latent variable and condition
            z = torch.randn(1, model.latent_size).to(device)
            z_cat = torch.cat((z, label_embedding), dim=1)
            
            # Generate structure tensor through decoder
            raw_output = model.decoder(z_cat)[0].cpu()  # [12, H, W]
            
            # Get class predictions for each layer (4 layers × 3 classes)
            predicted_tensor = torch.zeros(4, raw_output.size(1), raw_output.size(2), dtype=torch.long)
            
            for z in range(4):  # 4 layers
                layer_logits = raw_output[z*3:(z+1)*3]  # [3, H, W]
                # Move channel dimension to batch dimension [3, H, W] → [H, W, 3]
                layer_logits = layer_logits.permute(1, 2, 0)  
                # Select class with highest probability at each position
                _, predicted = torch.max(layer_logits, dim=2)
                predicted_tensor[z] = predicted
            
            # Visualize generated tensor
            if op in [0.2, 0.4, 0.6]:  # Only visualize representative values
                visualize_layers(predicted_tensor, result_dir, op)
            
            all_tensors.append(predicted_tensor)
            
            # Analyze tensor
            stats = analyze_tensor(predicted_tensor)
            all_value_counts.append(stats['value_counts'])
            
            # Write results to text file
            f.write(f"Overpotential: {op:.2f}V\n")
            f.write(f"Overall mean: {stats['global_mean']:.4f}\n")
            f.write(f"Overall standard deviation: {stats['global_std']:.4f}\n")
            f.write("Element distribution (class: count):\n")
            f.write(f"  Class 0 (Empty): {stats['value_counts'][0]} elements\n")
            f.write(f"  Class 1 (Pd): {stats['value_counts'][1]} elements\n")
            f.write(f"  Class 2 (Pt): {stats['value_counts'][2]} elements\n")
            
            # Display even and odd layers separately
            f.write("\nEven layer statistics (layers 1, 3, 5, ...):\n")
            for i, layer_stat in enumerate(stats['even_layers_stats']):
                layer_idx = i * 2
                f.write(f"  Layer {layer_idx+1}:\n")
                f.write(f"    Mean: {layer_stat['mean']:.4f}\n")
                f.write(f"    Standard deviation: {layer_stat['std']:.4f}\n")
                f.write(f"    Min: {layer_stat['min']:.4f}\n")
                f.write(f"    Max: {layer_stat['max']:.4f}\n")
                f.write(f"    Median: {layer_stat['median']:.4f}\n")
                f.write("    Element distribution (class: count):\n")
                f.write(f"      Class 0 (Empty): {layer_stat['id_counts'][0]} elements\n")
                f.write(f"      Class 1 (Pd): {layer_stat['id_counts'][1]} elements\n")
                f.write(f"      Class 2 (Pt): {layer_stat['id_counts'][2]} elements\n")
            
            f.write("\nOdd layer statistics (layers 2, 4, 6, ...):\n")
            for i, layer_stat in enumerate(stats['odd_layers_stats']):
                layer_idx = i * 2 + 1
                f.write(f"  Layer {layer_idx+1}:\n")
                f.write(f"    Mean: {layer_stat['mean']:.4f}\n")
                f.write(f"    Standard deviation: {layer_stat['std']:.4f}\n")
                f.write(f"    Min: {layer_stat['min']:.4f}\n")
                f.write(f"    Max: {layer_stat['max']:.4f}\n")
                f.write(f"    Median: {layer_stat['median']:.4f}\n")
                f.write("    Element distribution (class: count):\n")
                f.write(f"      Class 0 (Empty): {layer_stat['id_counts'][0]} elements\n")
                f.write(f"      Class 1 (Pd): {layer_stat['id_counts'][1]} elements\n")
                f.write(f"      Class 2 (Pt): {layer_stat['id_counts'][2]} elements\n")
            
            # Output tensor values (converted for visualization)
            viz_tensor = stats['viz_tensor']
            f.write("\nTensor values (converted for visualization):\n")
            for layer_idx in range(viz_tensor.size(0)):
                f.write(f"  Layer {layer_idx+1} ({'Even layer' if layer_idx % 2 == 0 else 'Odd layer'}):\n")
                layer_data = viz_tensor[layer_idx].numpy()
                # Set NumPy array display options
                np.set_printoptions(precision=3, suppress=True, threshold=1000)
                # Convert each row to string and write it out
                for row_idx in range(layer_data.shape[0]):
                    row_str = "    " + np.array2string(layer_data[row_idx], precision=3, suppress_small=True)
                    f.write(f"{row_str}\n")
                f.write("\n")
            
            f.write("\n" + "-" * 40 + "\n\n")
        
        # Plot change in element distribution with overpotential
        plot_element_distribution(overpotentials, all_value_counts, result_dir)

def plot_element_distribution(overpotentials, value_counts, result_dir):
    """Plot change in element distribution with overpotential"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.array(overpotentials)
    y0 = np.array([counts.get(0, 0) for counts in value_counts])  # Empty (Class 0)
    y1 = np.array([counts.get(1, 0) for counts in value_counts])  # Pd (Class 1)
    y2 = np.array([counts.get(2, 0) for counts in value_counts])  # Pt (Class 2)
    
    ax.plot(x, y0, 'g--', marker='^', label='Empty (Class 0)', linewidth=1.5)
    ax.plot(x, y1, 'b-', marker='o', label='Pd (Class 1)', linewidth=2)
    ax.plot(x, y2, 'r-', marker='s', label='Pt (Class 2)', linewidth=2)
    
    ax.set_xlabel('Overpotential (V)', fontsize=12)
    ax.set_ylabel('Number of Elements', fontsize=12)
    ax.set_title('Element Distribution by Overpotential', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # tight_layoutの代わりにsubplots_adjustを使用
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(result_dir / f"element_distribution.png")
    plt.close()

def visualize_layers(tensor, result_dir, overpotential):
    """Visualize structure of each layer"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # カラーマップの設定 (非推奨警告修正)
    cmap = 'viridis'  # 3クラス用のカラーマップ
    
    for i in range(4):
        ax = axes[i]
        layer_data = tensor[i].numpy()
        
        # 可視化
        im = ax.imshow(layer_data, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f"Layer {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # カラーバーを追加
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[0, 1, 2])
    cbar.set_ticklabels(['Empty', 'Pd', 'Pt'])
    
    plt.suptitle(f'Layer Structure at Overpotential {overpotential:.1f}V', fontsize=14)
    
    # tight_layoutの代わりにsubplots_adjustを使用
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)
    plt.savefig(result_dir / f"layer_structure_{overpotential:.1f}V.png")
    plt.close()

def main():
    """Main execution function"""
    print("Starting conditional VAE model test")
    
    # Check/create directory for saving results
    result_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully")
        
        # Test outputs with different conditions
        print("Starting output test with different overpotential conditions")
        test_conditional_outputs(model, result_dir)
        print("Output test completed")
        
        print(f"Results saved to {result_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()