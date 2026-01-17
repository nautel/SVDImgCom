"""
Module: visualization.py
Mo ta: Cac ham visualize ket qua nen anh va phan tich SVD.

Author: Student - KHTN University
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def plot_singular_values(S, title="Singular Value Spectrum", save_path=None):
    """
    Ve bieu do singular values.

    Parameters:
    -----------
    S : numpy.ndarray
        Mang singular values
    title : str
        Tieu de bieu do
    save_path : str, optional
        Duong dan luu file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(range(1, len(S)+1), S, linewidth=2, color='steelblue')
    ax.set_xlabel('Index i', fontsize=12)
    ax.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cumulative_energy(S, thresholds=[90, 95, 99], save_path=None):
    """
    Ve bieu do cumulative energy.

    Parameters:
    -----------
    S : numpy.ndarray
        Singular values
    thresholds : list
        Cac nguong % de ve duong ngang
    save_path : str, optional
        Duong dan luu file
    """
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(S)+1), cumulative_energy, linewidth=2, color='darkgreen')

    colors = ['red', 'orange', 'purple']
    for threshold, color in zip(thresholds, colors):
        ax.axhline(y=threshold, color=color, linestyle='--',
                   linewidth=1.5, label=f'{threshold}%')

    ax.set_xlabel('Number of Components k', fontsize=12)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax.set_title('Energy Preservation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 102])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_svd_matrices(U, S, Vt, save_path=None):
    """
    Ve heatmap cua cac ma tran U, Sigma, V^T.

    Parameters:
    -----------
    U : numpy.ndarray
        Left singular vectors
    S : numpy.ndarray
        Singular values
    Vt : numpy.ndarray
        Right singular vectors
    save_path : str, optional
        Duong dan luu file
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # U matrix
    im1 = axes[0].imshow(U, cmap='coolwarm', aspect='auto')
    axes[0].set_title('U Matrix (Left Singular Vectors)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Sigma matrix (diagonal)
    Sigma = np.zeros((len(S), len(S)))
    np.fill_diagonal(Sigma, S)
    im2 = axes[1].imshow(Sigma, cmap='hot', aspect='auto')
    axes[1].set_title('Σ Matrix (Singular Values)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Vt matrix
    im3 = axes[2].imshow(Vt, cmap='coolwarm', aspect='auto')
    axes[2].set_title('V^T Matrix (Right Singular Vectors)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_compression_comparison(original, compressed_dict, k_values, save_path=None):
    """
    Ve luoi so sanh anh goc va cac anh nen.

    Parameters:
    -----------
    original : numpy.ndarray
        Anh goc
    compressed_dict : dict
        Dictionary {k: compressed_image}
    k_values : list
        List cac gia tri k
    save_path : str, optional
        Duong dan luu file
    """
    n_images = len(k_values) + 1
    ncols = 3
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten()

    # Original
    if len(original.shape) == 2:
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    else:
        axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Compressed
    for idx, k in enumerate(k_values, start=1):
        compressed = compressed_dict[k]['image']
        psnr = compressed_dict[k].get('psnr', 0)
        saved = compressed_dict[k].get('space_saved', 0)

        if len(compressed.shape) == 2:
            axes[idx].imshow(compressed, cmap='gray', vmin=0, vmax=255)
        else:
            axes[idx].imshow(compressed)

        axes[idx].set_title(f'k={k}\nPSNR={psnr:.1f}dB, Saved={saved:.0f}%',
                           fontsize=11, fontweight='bold')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(k_values)+1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_maps(original, compressed_dict, k_values, save_path=None):
    """
    Ve ban do sai so (error maps).

    Parameters:
    -----------
    original : numpy.ndarray
        Anh goc
    compressed_dict : dict
        Dictionary {k: compressed_image}
    k_values : list
        List cac gia tri k
    save_path : str, optional
        Duong dan luu file
    """
    n_images = len(k_values) + 1
    ncols = 3
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten()

    # Title
    axes[0].text(0.5, 0.5, 'Error Maps\n|Original - Compressed|',
                ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Error maps
    for idx, k in enumerate(k_values, start=1):
        compressed = compressed_dict[k]['image']

        # Calculate error
        if len(original.shape) == 2:
            error = np.abs(original.astype(np.float64) - compressed.astype(np.float64))
        else:
            error = np.mean(np.abs(original.astype(np.float64) - compressed.astype(np.float64)), axis=2)

        im = axes[idx].imshow(error, cmap='hot', vmin=0, vmax=50)
        axes[idx].set_title(f'k={k}\nMax Error={error.max():.1f}',
                           fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    # Hide unused
    for idx in range(len(k_values)+1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_quality_vs_k(k_values, psnr_values, mse_values=None, ssim_values=None, save_path=None):
    """
    Ve bieu do chat luong theo k.

    Parameters:
    -----------
    k_values : list
        Cac gia tri k
    psnr_values : list
        Cac gia tri PSNR
    mse_values : list, optional
        Cac gia tri MSE
    ssim_values : list, optional
        Cac gia tri SSIM
    save_path : str, optional
        Duong dan luu file
    """
    if mse_values and ssim_values:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # PSNR
        axes[0, 0].plot(k_values, psnr_values, marker='o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=30, color='red', linestyle='--', label='Good (30 dB)')
        axes[0, 0].axhline(y=40, color='green', linestyle='--', label='Excellent (40 dB)')
        axes[0, 0].set_xlabel('k', fontsize=11)
        axes[0, 0].set_ylabel('PSNR (dB)', fontsize=11)
        axes[0, 0].set_title('PSNR vs k', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE
        axes[0, 1].plot(k_values, mse_values, marker='s', linewidth=2, markersize=6, color='orangered')
        axes[0, 1].set_xlabel('k', fontsize=11)
        axes[0, 1].set_ylabel('MSE', fontsize=11)
        axes[0, 1].set_title('MSE vs k', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # SSIM
        axes[1, 0].plot(k_values, ssim_values, marker='^', linewidth=2, markersize=6, color='purple')
        axes[1, 0].set_xlabel('k', fontsize=11)
        axes[1, 0].set_ylabel('SSIM', fontsize=11)
        axes[1, 0].set_title('SSIM vs k', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].grid(True, alpha=0.3)

        # Combined
        ax2 = axes[1, 1]
        ax2.plot(k_values, psnr_values, marker='o', label='PSNR', linewidth=2)
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('PSNR (dB)', fontsize=11, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax3 = ax2.twinx()
        ax3.plot(k_values, ssim_values, marker='^', color='purple', label='SSIM', linewidth=2)
        ax3.set_ylabel('SSIM', fontsize=11, color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.set_ylim([0, 1.05])

        axes[1, 1].set_title('PSNR & SSIM vs k', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, psnr_values, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=30, color='red', linestyle='--', label='Good (30 dB)')
        ax.axhline(y=40, color='green', linestyle='--', label='Excellent (40 dB)')
        ax.set_xlabel('k (number of singular values)', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('Image Quality vs Compression Level', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_trade_off(compression_ratios, psnr_values, save_path=None):
    """
    Ve trade-off giua compression ratio va quality.

    Parameters:
    -----------
    compression_ratios : list
        % space saved
    psnr_values : list
        PSNR values
    save_path : str, optional
        Duong dan luu file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(compression_ratios, psnr_values, marker='D', linewidth=2,
            markersize=6, color='darkgreen')
    ax.set_xlabel('Space Saved (%)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Trade-off: Quality vs Compression', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    """Test visualization functions"""
    print("Testing visualization.py...")

    # Create test data
    print("\n1. Creating test data...")
    np.random.seed(42)
    S = np.sort(np.random.rand(100) * 1000)[::-1]
    print(f"   Singular values: {len(S)}")

    # Test plot_singular_values
    print("\n2. Testing plot_singular_values...")
    plot_singular_values(S)
    print("   OK")

    # Test plot_cumulative_energy
    print("\n3. Testing plot_cumulative_energy...")
    plot_cumulative_energy(S)
    print("   OK")

    # Test plot_quality_vs_k
    print("\n4. Testing plot_quality_vs_k...")
    k_vals = [5, 10, 20, 50, 100]
    psnr_vals = [25, 30, 35, 40, 45]
    mse_vals = [100, 50, 25, 10, 5]
    ssim_vals = [0.85, 0.90, 0.95, 0.98, 0.99]
    plot_quality_vs_k(k_vals, psnr_vals, mse_vals, ssim_vals)
    print("   OK")

    print("\nAll tests passed!")
