# -*- coding: utf-8 -*-
"""
Test SVD Compression with User's Real Images
"""
import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from image_utils import load_image, save_image, get_image_info
from svd_compression import (compress_grayscale, compress_rgb, get_svd_matrices,
                             calculate_compression_ratio, calculate_cumulative_energy)
from quality_metrics import calculate_all_metrics
from visualization import (plot_singular_values, plot_cumulative_energy,
                           plot_compression_comparison, plot_error_maps,
                           plot_quality_vs_k, plot_trade_off)

print("="*80)
print("SVD IMAGE COMPRESSION - Testing with Real Images")
print("="*80)

# ==================== TEST GRAYSCALE IMAGE ====================
print("\n" + "="*80)
print("PART 1: GRAYSCALE IMAGE COMPRESSION")
print("="*80)

# Load grayscale image
print("\n[1] Loading grayscale image...")
gray_path = 'images/grayscale/5.3.01.tiff'
try:
    img_gray = load_image(gray_path, mode='GRAY')
    print(f"   Loaded: {gray_path}")
    info = get_image_info(img_gray)
    print(f"   Shape: {info['shape']}, dtype: {info['dtype']}")
    print(f"   Min: {info['min']}, Max: {info['max']}, Mean: {info['mean']:.1f}")
except Exception as e:
    print(f"   Error loading image: {e}")
    print("   Skipping grayscale test...")
    img_gray = None

if img_gray is not None:
    # SVD Analysis
    print("\n[2] SVD Analysis...")
    U, S, Vt = get_svd_matrices(img_gray)
    print(f"   U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
    print(f"   Top 5 singular values: {S[:5]}")

    cumulative_energy = calculate_cumulative_energy(S)
    k_90 = np.argmax(cumulative_energy >= 90) + 1
    k_95 = np.argmax(cumulative_energy >= 95) + 1
    k_99 = np.argmax(cumulative_energy >= 99) + 1
    print(f"   For 90% energy: k = {k_90}")
    print(f"   For 95% energy: k = {k_95}")
    print(f"   For 99% energy: k = {k_99}")

    # Visualize SVD
    plot_singular_values(S, "Grayscale - Singular Value Spectrum",
                        save_path='results/visualizations/gray_singular_values.png')
    plot_cumulative_energy(S, save_path='results/visualizations/gray_cumulative_energy.png')

    # Compression with different k values
    print("\n[3] Testing compression...")
    k_values = [5, 10, 20, 50, 100, 150]
    # Adjust k_values to not exceed image dimensions
    k_values = [k for k in k_values if k <= min(img_gray.shape)]

    print(f"\n{'k':>5} | {'PSNR':>8} | {'MSE':>10} | {'SSIM':>8} | {'Saved':>8}")
    print("-" * 55)

    results_gray = {}
    for k in k_values:
        compressed = compress_grayscale(img_gray, k)
        metrics = calculate_all_metrics(img_gray, compressed, include_ssim=True)
        stats = calculate_compression_ratio(img_gray.shape, k, is_rgb=False)

        results_gray[k] = {
            'image': compressed,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'ssim': metrics['ssim'],
            'space_saved': stats['space_saved_percent']
        }

        print(f"{k:5d} | {metrics['psnr']:8.2f} | {metrics['mse']:10.2f} | "
              f"{metrics['ssim']:8.4f} | {stats['space_saved_percent']:7.1f}%")

        save_image(compressed, f'results/compressed/gray_k{k:03d}.png')

    # Find optimal k
    optimal_k_gray = None
    for k in k_values:
        if results_gray[k]['psnr'] >= 30:
            optimal_k_gray = k
            break
    if optimal_k_gray is None:
        optimal_k_gray = k_values[-1]

    print(f"\n   Optimal k (PSNR >= 30dB): {optimal_k_gray}")
    print(f"   PSNR: {results_gray[optimal_k_gray]['psnr']:.2f} dB")
    print(f"   Space saved: {results_gray[optimal_k_gray]['space_saved']:.1f}%")

    # Visualizations
    print("\n[4] Creating visualizations...")
    plot_compression_comparison(img_gray, results_gray, k_values,
                               save_path='results/visualizations/gray_comparison.png')
    plot_error_maps(img_gray, results_gray, k_values,
                   save_path='results/visualizations/gray_error_maps.png')

    # Quality vs k
    k_range = list(range(5, min(img_gray.shape), 10))
    psnr_list, mse_list, ssim_list, saved_list = [], [], [], []
    for k in k_range:
        comp = compress_grayscale(img_gray, k)
        m = calculate_all_metrics(img_gray, comp, include_ssim=True)
        s = calculate_compression_ratio(img_gray.shape, k, is_rgb=False)
        psnr_list.append(m['psnr'])
        mse_list.append(m['mse'])
        ssim_list.append(m['ssim'])
        saved_list.append(s['space_saved_percent'])

    plot_quality_vs_k(k_range, psnr_list, mse_list, ssim_list,
                     save_path='results/visualizations/gray_quality_vs_k.png')
    plot_trade_off(saved_list, psnr_list,
                  save_path='results/visualizations/gray_tradeoff.png')

    print("   All grayscale visualizations saved!")

# ==================== TEST COLOR IMAGE ====================
print("\n" + "="*80)
print("PART 2: COLOR (RGB) IMAGE COMPRESSION")
print("="*80)

# Load color image
print("\n[1] Loading color image...")
color_path = 'images/color/4.1.01.tiff'
try:
    img_color = load_image(color_path, mode='RGB')
    print(f"   Loaded: {color_path}")
    info = get_image_info(img_color)
    print(f"   Shape: {info['shape']}, dtype: {info['dtype']}")
    print(f"   Min: {info['min']}, Max: {info['max']}, Mean: {info['mean']:.1f}")
except Exception as e:
    print(f"   Error loading image: {e}")
    print("   Skipping color test...")
    img_color = None

if img_color is not None:
    # Compression with different k values
    print("\n[2] Testing RGB compression...")
    k_values_rgb = [5, 10, 20, 50, 100]
    k_values_rgb = [k for k in k_values_rgb if k <= min(img_color.shape[:2])]

    print(f"\n{'k':>5} | {'PSNR':>8} | {'MSE':>10} | {'Saved':>8}")
    print("-" * 45)

    results_color = {}
    for k in k_values_rgb:
        compressed = compress_rgb(img_color, k)
        metrics = calculate_all_metrics(img_color, compressed, include_ssim=False)
        stats = calculate_compression_ratio(img_color.shape, k, is_rgb=True)

        results_color[k] = {
            'image': compressed,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'space_saved': stats['space_saved_percent']
        }

        print(f"{k:5d} | {metrics['psnr']:8.2f} | {metrics['mse']:10.2f} | "
              f"{stats['space_saved_percent']:7.1f}%")

        save_image(compressed, f'results/compressed/color_k{k:03d}.png')

    # Find optimal k
    optimal_k_color = None
    for k in k_values_rgb:
        if results_color[k]['psnr'] >= 30:
            optimal_k_color = k
            break
    if optimal_k_color is None:
        optimal_k_color = k_values_rgb[-1]

    print(f"\n   Optimal k (PSNR >= 30dB): {optimal_k_color}")
    print(f"   PSNR: {results_color[optimal_k_color]['psnr']:.2f} dB")
    print(f"   Space saved: {results_color[optimal_k_color]['space_saved']:.1f}%")

    # Visualizations
    print("\n[3] Creating visualizations...")
    plot_compression_comparison(img_color, results_color, k_values_rgb,
                               save_path='results/visualizations/color_comparison.png')
    plot_error_maps(img_color, results_color, k_values_rgb,
                   save_path='results/visualizations/color_error_maps.png')

    # Quality vs k
    k_range_rgb = list(range(5, min(img_color.shape[:2]), 10))
    psnr_list_rgb, saved_list_rgb = [], []
    for k in k_range_rgb:
        comp = compress_rgb(img_color, k)
        m = calculate_all_metrics(img_color, comp)
        s = calculate_compression_ratio(img_color.shape, k, is_rgb=True)
        psnr_list_rgb.append(m['psnr'])
        saved_list_rgb.append(s['space_saved_percent'])

    plot_quality_vs_k(k_range_rgb, psnr_list_rgb,
                     save_path='results/visualizations/color_quality_vs_k.png')
    plot_trade_off(saved_list_rgb, psnr_list_rgb,
                  save_path='results/visualizations/color_tradeoff.png')

    print("   All color visualizations saved!")

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if img_gray is not None:
    print(f"\nGrayscale Image:")
    print(f"  Shape: {img_gray.shape}")
    print(f"  Optimal k: {optimal_k_gray}")
    print(f"  Best PSNR: {results_gray[optimal_k_gray]['psnr']:.2f} dB")
    print(f"  Space saved: {results_gray[optimal_k_gray]['space_saved']:.1f}%")

if img_color is not None:
    print(f"\nColor Image:")
    print(f"  Shape: {img_color.shape}")
    print(f"  Optimal k: {optimal_k_color}")
    print(f"  Best PSNR: {results_color[optimal_k_color]['psnr']:.2f} dB")
    print(f"  Space saved: {results_color[optimal_k_color]['space_saved']:.1f}%")

print("\n" + "="*80)
print("ALL TESTS COMPLETED!")
print("="*80)
print("\nResults saved to:")
print("  - Compressed images: results/compressed/")
print("  - Visualizations: results/visualizations/")
print("\nGenerated files:")
if img_gray is not None:
    print("  Grayscale:")
    print("    - gray_k*.png (compressed images)")
    print("    - gray_*.png (visualizations)")
if img_color is not None:
    print("  Color:")
    print("    - color_k*.png (compressed images)")
    print("    - color_*.png (visualizations)")
print("\nDone! Check the results folder.")
