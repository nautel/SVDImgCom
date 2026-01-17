# -*- coding: utf-8 -*-
"""
Export High-Resolution Figures for Report
This script creates publication-quality figures for the final report.
"""
import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from image_utils import load_image
from svd_compression import compress_grayscale, compress_rgb, get_svd_matrices, calculate_cumulative_energy
from quality_metrics import calculate_all_metrics
from svd_compression import calculate_compression_ratio

# Configuration
FIGURE_DPI = 300  # High resolution for report
OUTPUT_DIR = Path('report/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

print("="*80)
print("EXPORTING HIGH-RESOLUTION FIGURES FOR REPORT")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"DPI: {FIGURE_DPI}")
print()

# ============================================================================
# Load Images
# ============================================================================
print("[1] Loading images...")
img_gray = load_image('images/grayscale/5.3.01.tiff', mode='GRAY')
img_rgb = load_image('images/color/4.1.01.tiff', mode='RGB')
print(f"    Grayscale: {img_gray.shape}")
print(f"    RGB: {img_rgb.shape}")

# ============================================================================
# Figure 1: Original Test Images
# ============================================================================
print("\n[2] Creating Figure 1: Original Test Images...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('(a) Grayscale Test Image\n512×512 pixels', fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(img_rgb)
axes[1].set_title('(b) RGB Test Image\n512×512×3 pixels', fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_original_images.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig1_original_images.png")

# ============================================================================
# Figure 2: SVD Decomposition Concept (Grayscale)
# ============================================================================
print("\n[3] Creating Figure 2: SVD Decomposition...")
U, S, Vt = get_svd_matrices(img_gray)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
axes[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('(a) Original Image A\n(512×512)', fontsize=10, fontweight='bold')
axes[0].axis('off')

# U matrix
im1 = axes[1].imshow(U[:, :100], cmap='RdBu_r', aspect='auto')
axes[1].set_title('(b) U Matrix\n(512×512)', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Column index', fontsize=9)
axes[1].set_ylabel('Row index', fontsize=9)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Sigma (diagonal)
Sigma_diag = np.zeros((100, 100))
np.fill_diagonal(Sigma_diag, S[:100])
im2 = axes[2].imshow(Sigma_diag, cmap='hot', aspect='auto')
axes[2].set_title('(c) Σ Matrix (top 100×100)\nSingular Values', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Column index', fontsize=9)
axes[2].set_ylabel('Row index', fontsize=9)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# V^T matrix
im3 = axes[3].imshow(Vt[:100, :], cmap='RdBu_r', aspect='auto')
axes[3].set_title('(d) V^T Matrix\n(512×512)', fontsize=10, fontweight='bold')
axes[3].set_xlabel('Column index', fontsize=9)
axes[3].set_ylabel('Row index', fontsize=9)
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_svd_decomposition.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig2_svd_decomposition.png")

# ============================================================================
# Figure 3: Singular Value Spectrum
# ============================================================================
print("\n[4] Creating Figure 3: Singular Value Spectrum...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Grayscale spectrum
U_g, S_g, Vt_g = get_svd_matrices(img_gray)
axes[0].semilogy(range(1, len(S_g)+1), S_g, linewidth=2, color='steelblue')
axes[0].set_xlabel('Index i', fontsize=11)
axes[0].set_ylabel('Singular Value σᵢ (log scale)', fontsize=11)
axes[0].set_title('(a) Grayscale Image\nSingular Value Decay', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 512])

# RGB spectrum (all channels)
R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
_, S_r, _ = get_svd_matrices(R)
_, S_g_rgb, _ = get_svd_matrices(G)
_, S_b, _ = get_svd_matrices(B)

axes[1].semilogy(range(1, len(S_r)+1), S_r, linewidth=2, color='red', alpha=0.7, label='Red')
axes[1].semilogy(range(1, len(S_g_rgb)+1), S_g_rgb, linewidth=2, color='green', alpha=0.7, label='Green')
axes[1].semilogy(range(1, len(S_b)+1), S_b, linewidth=2, color='blue', alpha=0.7, label='Blue')
axes[1].set_xlabel('Index i', fontsize=11)
axes[1].set_ylabel('Singular Value σᵢ (log scale)', fontsize=11)
axes[1].set_title('(b) RGB Image\nPer-Channel Singular Values', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 512])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_singular_value_spectrum.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig3_singular_value_spectrum.png")

# ============================================================================
# Figure 4: Cumulative Energy
# ============================================================================
print("\n[5] Creating Figure 4: Cumulative Energy...")
fig, ax = plt.subplots(figsize=(10, 6))

# Grayscale
cumulative_energy_gray = calculate_cumulative_energy(S_g)
ax.plot(range(1, len(cumulative_energy_gray)+1), cumulative_energy_gray,
        linewidth=2.5, label='Grayscale', color='gray')

# RGB (average)
cumulative_energy_r = calculate_cumulative_energy(S_r)
cumulative_energy_g = calculate_cumulative_energy(S_g_rgb)
cumulative_energy_b = calculate_cumulative_energy(S_b)
cumulative_energy_rgb = (cumulative_energy_r + cumulative_energy_g + cumulative_energy_b) / 3
ax.plot(range(1, len(cumulative_energy_rgb)+1), cumulative_energy_rgb,
        linewidth=2.5, label='RGB (average)', color='steelblue', linestyle='--')

# Threshold lines
ax.axhline(y=90, color='red', linestyle=':', linewidth=2, label='90% energy', alpha=0.7)
ax.axhline(y=95, color='orange', linestyle=':', linewidth=2, label='95% energy', alpha=0.7)
ax.axhline(y=99, color='purple', linestyle=':', linewidth=2, label='99% energy', alpha=0.7)

# Find k for 90%
k_90_gray = np.argmax(cumulative_energy_gray >= 90) + 1
k_90_rgb = np.argmax(cumulative_energy_rgb >= 90) + 1
ax.axvline(x=k_90_gray, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=k_90_rgb, color='steelblue', linestyle='--', alpha=0.5)

ax.text(k_90_gray, 85, f'k={k_90_gray}', ha='center', fontsize=9)
ax.text(k_90_rgb, 92, f'k={k_90_rgb}', ha='center', fontsize=9)

ax.set_xlabel('Number of Components k', fontsize=12)
ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
ax.set_title('Energy Preservation vs Number of Components', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 200])
ax.set_ylim([0, 102])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_cumulative_energy.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig4_cumulative_energy.png")

# ============================================================================
# Figure 5: Compression Results at Different k
# ============================================================================
print("\n[6] Creating Figure 5: Compression Results...")
k_values = [5, 10, 20, 50, 100]

# Grayscale
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('(a) Original\nGrayscale', fontsize=10, fontweight='bold')
axes[0].axis('off')

for idx, k in enumerate(k_values, start=1):
    compressed = compress_grayscale(img_gray, k)
    metrics = calculate_all_metrics(img_gray, compressed)
    stats = calculate_compression_ratio(img_gray.shape, k, is_rgb=False)

    axes[idx].imshow(compressed, cmap='gray', vmin=0, vmax=255)
    axes[idx].set_title(f'({chr(97+idx)}) k={k}\n'
                       f'PSNR={metrics["psnr"]:.1f}dB, '
                       f'Saved={stats["space_saved_percent"]:.0f}%',
                       fontsize=9, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_grayscale_compression.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig5_grayscale_compression.png")

# RGB
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img_rgb)
axes[0].set_title('(a) Original\nRGB', fontsize=10, fontweight='bold')
axes[0].axis('off')

for idx, k in enumerate(k_values, start=1):
    compressed = compress_rgb(img_rgb, k)
    metrics = calculate_all_metrics(img_rgb, compressed)
    stats = calculate_compression_ratio(img_rgb.shape, k, is_rgb=True)

    axes[idx].imshow(compressed)
    axes[idx].set_title(f'({chr(97+idx)}) k={k}\n'
                       f'PSNR={metrics["psnr"]:.1f}dB, '
                       f'Saved={stats["space_saved_percent"]:.0f}%',
                       fontsize=9, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig6_rgb_compression.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig6_rgb_compression.png")

# ============================================================================
# Figure 7: Quality Metrics vs k
# ============================================================================
print("\n[7] Creating Figure 7: Quality Metrics vs k...")
k_range = list(range(5, min(img_gray.shape), 15))

# Grayscale metrics
psnr_gray, mse_gray, ssim_gray = [], [], []
for k in k_range:
    comp = compress_grayscale(img_gray, k)
    m = calculate_all_metrics(img_gray, comp, include_ssim=True)
    psnr_gray.append(m['psnr'])
    mse_gray.append(m['mse'])
    ssim_gray.append(m['ssim'])

# RGB metrics
psnr_rgb, mse_rgb = [], []
for k in k_range:
    comp = compress_rgb(img_rgb, k)
    m = calculate_all_metrics(img_rgb, comp)
    psnr_rgb.append(m['psnr'])
    mse_rgb.append(m['mse'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PSNR
axes[0].plot(k_range, psnr_gray, marker='o', linewidth=2, markersize=5, label='Grayscale')
axes[0].plot(k_range, psnr_rgb, marker='s', linewidth=2, markersize=5, label='RGB')
axes[0].axhline(y=30, color='red', linestyle='--', linewidth=1.5, label='Good (30dB)', alpha=0.7)
axes[0].axhline(y=40, color='green', linestyle='--', linewidth=1.5, label='Excellent (40dB)', alpha=0.7)
axes[0].set_xlabel('k (number of components)', fontsize=11)
axes[0].set_ylabel('PSNR (dB)', fontsize=11)
axes[0].set_title('(a) Peak Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# MSE
axes[1].plot(k_range, mse_gray, marker='o', linewidth=2, markersize=5, label='Grayscale', color='orangered')
axes[1].plot(k_range, mse_rgb, marker='s', linewidth=2, markersize=5, label='RGB', color='darkred')
axes[1].set_xlabel('k (number of components)', fontsize=11)
axes[1].set_ylabel('MSE', fontsize=11)
axes[1].set_title('(b) Mean Squared Error', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# SSIM (grayscale only)
axes[2].plot(k_range, ssim_gray, marker='^', linewidth=2, markersize=5, color='purple')
axes[2].set_xlabel('k (number of components)', fontsize=11)
axes[2].set_ylabel('SSIM', fontsize=11)
axes[2].set_title('(c) Structural Similarity Index\n(Grayscale)', fontsize=11, fontweight='bold')
axes[2].set_ylim([0, 1.05])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_quality_metrics.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig7_quality_metrics.png")

# ============================================================================
# Figure 8: Trade-off Analysis
# ============================================================================
print("\n[8] Creating Figure 8: Trade-off Analysis...")
# Calculate compression ratios
saved_gray, saved_rgb = [], []
for k in k_range:
    stats_g = calculate_compression_ratio(img_gray.shape, k, is_rgb=False)
    stats_r = calculate_compression_ratio(img_rgb.shape, k, is_rgb=True)
    saved_gray.append(stats_g['space_saved_percent'])
    saved_rgb.append(stats_r['space_saved_percent'])

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(saved_gray, psnr_gray, marker='o', markersize=8, linewidth=2.5,
        label='Grayscale', color='gray')
ax.plot(saved_rgb, psnr_rgb, marker='s', markersize=8, linewidth=2.5,
        label='RGB', color='steelblue')

# Annotate some key points
for i in [0, len(k_range)//2, -1]:
    ax.annotate(f'k={k_range[i]}',
                (saved_gray[i], psnr_gray[i]),
                textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9, color='gray')
    ax.annotate(f'k={k_range[i]}',
                (saved_rgb[i], psnr_rgb[i]),
                textcoords="offset points", xytext=(0,-15),
                ha='center', fontsize=9, color='steelblue')

ax.axhline(y=30, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Target PSNR (30dB)')
ax.set_xlabel('Space Saved (%)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('Quality vs Compression Trade-off', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_tradeoff.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig8_tradeoff.png")

# ============================================================================
# Figure 9: Error Maps
# ============================================================================
print("\n[9] Creating Figure 9: Error Maps...")
k_test = [10, 20, 50]

fig, axes = plt.subplots(2, len(k_test), figsize=(12, 8))

for idx, k in enumerate(k_test):
    # Grayscale error
    comp_g = compress_grayscale(img_gray, k)
    error_g = np.abs(img_gray.astype(np.float64) - comp_g.astype(np.float64))

    im1 = axes[0, idx].imshow(error_g, cmap='hot', vmin=0, vmax=50)
    axes[0, idx].set_title(f'({chr(97+idx)}) k={k}\nMax={error_g.max():.1f}',
                          fontsize=10, fontweight='bold')
    axes[0, idx].axis('off')
    plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)

    # RGB error
    comp_r = compress_rgb(img_rgb, k)
    error_r = np.mean(np.abs(img_rgb.astype(np.float64) - comp_r.astype(np.float64)), axis=2)

    im2 = axes[1, idx].imshow(error_r, cmap='hot', vmin=0, vmax=50)
    axes[1, idx].set_title(f'({chr(97+idx+3)}) k={k}\nMax={error_r.max():.1f}',
                          fontsize=10, fontweight='bold')
    axes[1, idx].axis('off')
    plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)

axes[0, 0].set_ylabel('Grayscale', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('RGB', fontsize=11, fontweight='bold')

plt.suptitle('Error Maps: |Original - Compressed|', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig9_error_maps.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print("    Saved: fig9_error_maps.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("EXPORT COMPLETE!")
print("="*80)
print(f"\nAll figures saved to: {OUTPUT_DIR}")
print("\nGenerated figures:")
print("  1. fig1_original_images.png - Test images")
print("  2. fig2_svd_decomposition.png - SVD matrices U, Σ, V^T")
print("  3. fig3_singular_value_spectrum.png - Singular value decay")
print("  4. fig4_cumulative_energy.png - Energy preservation")
print("  5. fig5_grayscale_compression.png - Grayscale results")
print("  6. fig6_rgb_compression.png - RGB results")
print("  7. fig7_quality_metrics.png - PSNR, MSE, SSIM vs k")
print("  8. fig8_tradeoff.png - Quality vs compression trade-off")
print("  9. fig9_error_maps.png - Compression error visualization")
print(f"\nDPI: {FIGURE_DPI} (publication quality)")
print("="*80)
print("\nThese figures are ready to be inserted into the report!")
print("="*80)
