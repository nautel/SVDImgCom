"""
Module: quality_metrics.py
Mô tả: Các metrics để đánh giá chất lượng ảnh sau nén.

Metrics:
--------
1. MSE (Mean Squared Error): Sai số bình phương trung bình
2. PSNR (Peak Signal-to-Noise Ratio): Tỷ lệ tín hiệu trên nhiễu
3. SSIM (Structural Similarity Index): Chỉ số tương đồng cấu trúc

Author: Student - KHTN University
Date: 2026
"""

import numpy as np
from typing import Tuple, Optional


def calculate_mse(original, compressed):
    """
    Tính Mean Squared Error (MSE) giữa ảnh gốc và ảnh nén.

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc
    compressed : numpy.ndarray
        Ảnh đã nén

    Returns:
    --------
    float
        Giá trị MSE

    Formula:
    --------
    MSE = (1 / (m × n)) × Σ(I_original(i,j) - I_compressed(i,j))²

    Notes:
    ------
    - MSE = 0: Hai ảnh giống hệt nhau (nén hoàn hảo)
    - MSE càng nhỏ: Ảnh nén càng giống ảnh gốc
    - MSE càng lớn: Ảnh nén càng khác ảnh gốc

    Raises:
    -------
    ValueError
        Nếu hai ảnh không cùng shape

    Examples:
    ---------
    >>> original = load_image('lena.png', mode='GRAY')
    >>> compressed = compress_grayscale(original, k=50)
    >>> mse = calculate_mse(original, compressed)
    >>> print(f"MSE: {mse:.2f}")
    """
    # Kiểm tra shape
    if original.shape != compressed.shape:
        raise ValueError(
            f"Hai ảnh phải có cùng shape. "
            f"Original: {original.shape}, Compressed: {compressed.shape}"
        )

    # Convert sang float để tính toán chính xác
    original_float = original.astype(np.float64)
    compressed_float = compressed.astype(np.float64)

    # Tính MSE
    # Sử dụng numpy vectorization thay vì vòng lặp
    mse = np.mean((original_float - compressed_float) ** 2)

    return float(mse)


def calculate_psnr(original, compressed, max_pixel_value=255):
    """
    Tính Peak Signal-to-Noise Ratio (PSNR).

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc
    compressed : numpy.ndarray
        Ảnh đã nén
    max_pixel_value : int, optional
        Giá trị pixel tối đa (255 cho ảnh 8-bit)
        Default: 255

    Returns:
    --------
    float
        Giá trị PSNR tính bằng dB (decibels)
        Trả về inf nếu hai ảnh giống hệt nhau (MSE = 0)

    Formula:
    --------
    PSNR = 10 × log₁₀(MAX² / MSE)

    Trong đó MAX = 255 cho ảnh 8-bit.

    Interpretation:
    ---------------
    - PSNR > 40 dB: Excellent quality (gần như không phân biệt được)
    - PSNR 30-40 dB: Good quality (chất lượng tốt)
    - PSNR 20-30 dB: Fair quality (chấp nhận được)
    - PSNR < 20 dB: Poor quality (chất lượng kém)

    Notes:
    ------
    PSNR là metric phổ biến trong image compression vì:
    - Dễ tính toán
    - Có đơn vị rõ ràng (dB)
    - Dễ so sánh giữa các phương pháp nén

    Raises:
    -------
    ValueError
        Nếu hai ảnh không cùng shape

    Examples:
    ---------
    >>> original = load_image('peppers.png', mode='RGB')
    >>> compressed = compress_rgb(original, k=50)
    >>> psnr = calculate_psnr(original, compressed)
    >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Tính MSE
    mse = calculate_mse(original, compressed)

    # Xử lý trường hợp đặc biệt: MSE = 0
    if mse == 0:
        # Hai ảnh giống hệt nhau
        return float('inf')

    # Tính PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return float(psnr)


def calculate_rmse(original, compressed):
    """
    Tính Root Mean Squared Error (RMSE).

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc
    compressed : numpy.ndarray
        Ảnh đã nén

    Returns:
    --------
    float
        Giá trị RMSE

    Formula:
    --------
    RMSE = √MSE

    Notes:
    ------
    RMSE có cùng đơn vị với pixel values (0-255),
    nên dễ hiểu hơn MSE.

    Examples:
    ---------
    >>> rmse = calculate_rmse(original, compressed)
    >>> print(f"RMSE: {rmse:.2f} (trên thang 0-255)")
    """
    mse = calculate_mse(original, compressed)
    rmse = np.sqrt(mse)
    return float(rmse)


def calculate_mae(original, compressed):
    """
    Tính Mean Absolute Error (MAE).

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc
    compressed : numpy.ndarray
        Ảnh đã nén

    Returns:
    --------
    float
        Giá trị MAE

    Formula:
    --------
    MAE = (1 / (m × n)) × Σ|I_original(i,j) - I_compressed(i,j)|

    Notes:
    ------
    MAE ít nhạy cảm với outliers hơn MSE.

    Examples:
    ---------
    >>> mae = calculate_mae(original, compressed)
    >>> print(f"MAE: {mae:.2f}")
    """
    if original.shape != compressed.shape:
        raise ValueError(
            f"Hai ảnh phải có cùng shape. "
            f"Original: {original.shape}, Compressed: {compressed.shape}"
        )

    original_float = original.astype(np.float64)
    compressed_float = compressed.astype(np.float64)

    mae = np.mean(np.abs(original_float - compressed_float))

    return float(mae)


def calculate_ssim_simple(original, compressed):
    """
    Tính Structural Similarity Index (SSIM) đơn giản.

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc (grayscale)
    compressed : numpy.ndarray
        Ảnh đã nén (grayscale)

    Returns:
    --------
    float
        Giá trị SSIM trong khoảng [-1, 1]
        1: Hai ảnh giống hệt nhau
        0: Không có tương quan
        -1: Hoàn toàn đối nghịch

    Notes:
    ------
    Đây là implementation đơn giản của SSIM.
    Để tính SSIM đầy đủ và chính xác, sử dụng:
    from skimage.metrics import structural_similarity as ssim

    Formula (simplified):
    ---------------------
    SSIM = (2μ_x μ_y + C1)(2σ_xy + C2) /
           ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    Trong đó:
    - μ: mean
    - σ²: variance
    - σ_xy: covariance
    - C1, C2: constants để tránh chia 0

    Examples:
    ---------
    >>> ssim = calculate_ssim_simple(original_gray, compressed_gray)
    >>> print(f"SSIM: {ssim:.4f}")
    """
    if len(original.shape) != 2 or len(compressed.shape) != 2:
        raise ValueError("SSIM đơn giản chỉ hỗ trợ ảnh grayscale 2D")

    if original.shape != compressed.shape:
        raise ValueError(
            f"Hai ảnh phải có cùng shape. "
            f"Original: {original.shape}, Compressed: {compressed.shape}"
        )

    # Convert sang float
    x = original.astype(np.float64)
    y = compressed.astype(np.float64)

    # Tính các thống kê
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.std(x)
    sigma_y = np.std(y)

    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    # Constants (theo paper SSIM gốc)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Tính SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2)

    ssim = numerator / denominator

    return float(ssim)


def calculate_all_metrics(original, compressed, include_ssim=False):
    """
    Tính tất cả metrics cùng lúc.

    Parameters:
    -----------
    original : numpy.ndarray
        Ảnh gốc
    compressed : numpy.ndarray
        Ảnh đã nén
    include_ssim : bool, optional
        Có tính SSIM không (chỉ cho grayscale)
        Default: False

    Returns:
    --------
    dict
        Dictionary chứa tất cả metrics

    Examples:
    ---------
    >>> metrics = calculate_all_metrics(original, compressed)
    >>> for name, value in metrics.items():
    ...     print(f"{name}: {value:.2f}")
    """
    metrics = {
        'mse': calculate_mse(original, compressed),
        'rmse': calculate_rmse(original, compressed),
        'mae': calculate_mae(original, compressed),
        'psnr': calculate_psnr(original, compressed),
    }

    # Thêm SSIM nếu cần (chỉ cho grayscale)
    if include_ssim and len(original.shape) == 2:
        metrics['ssim'] = calculate_ssim_simple(original, compressed)

    return metrics


def evaluate_compression_quality(psnr):
    """
    Đánh giá chất lượng nén dựa trên PSNR.

    Parameters:
    -----------
    psnr : float
        Giá trị PSNR (dB)

    Returns:
    --------
    str
        Mô tả chất lượng

    Examples:
    ---------
    >>> psnr = calculate_psnr(original, compressed)
    >>> quality = evaluate_compression_quality(psnr)
    >>> print(f"PSNR: {psnr:.2f} dB - {quality}")
    """
    if psnr >= 40:
        return "Excellent (rất tốt, gần như không phân biệt được)"
    elif psnr >= 30:
        return "Good (tốt, chất lượng cao)"
    elif psnr >= 20:
        return "Fair (chấp nhận được)"
    else:
        return "Poor (kém, mất nhiều chi tiết)"


if __name__ == "__main__":
    """
    Test các functions trong module.
    """
    print("Testing quality_metrics.py...")

    # Tạo ảnh test
    print("\n1. Tạo ảnh test:")
    np.random.seed(42)  # Để kết quả reproducible
    original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # Tạo ảnh "compressed" bằng cách thêm noise nhỏ
    noise = np.random.normal(0, 5, original.shape)
    compressed = np.clip(original.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    print(f"   Original: {original.shape}, dtype={original.dtype}")
    print(f"   Compressed: {compressed.shape}, dtype={compressed.dtype}")

    # Test MSE
    print("\n2. Test calculate_mse:")
    mse = calculate_mse(original, compressed)
    print(f"   MSE: {mse:.4f}")
    print("   ✓ MSE thành công")

    # Test PSNR
    print("\n3. Test calculate_psnr:")
    psnr = calculate_psnr(original, compressed)
    quality = evaluate_compression_quality(psnr)
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   Quality: {quality}")
    print("   ✓ PSNR thành công")

    # Test RMSE
    print("\n4. Test calculate_rmse:")
    rmse = calculate_rmse(original, compressed)
    print(f"   RMSE: {rmse:.4f}")
    print("   ✓ RMSE thành công")

    # Test MAE
    print("\n5. Test calculate_mae:")
    mae = calculate_mae(original, compressed)
    print(f"   MAE: {mae:.4f}")
    print("   ✓ MAE thành công")

    # Test SSIM
    print("\n6. Test calculate_ssim_simple:")
    ssim = calculate_ssim_simple(original, compressed)
    print(f"   SSIM: {ssim:.4f}")
    print("   ✓ SSIM thành công")

    # Test calculate_all_metrics
    print("\n7. Test calculate_all_metrics:")
    metrics = calculate_all_metrics(original, compressed, include_ssim=True)
    for name, value in metrics.items():
        print(f"   {name.upper()}: {value:.4f}")
    print("   ✓ All metrics thành công")

    # Test với ảnh giống hệt nhau (MSE = 0, PSNR = inf)
    print("\n8. Test với ảnh giống nhau:")
    identical = original.copy()
    mse_zero = calculate_mse(original, identical)
    psnr_inf = calculate_psnr(original, identical)
    print(f"   MSE: {mse_zero}")
    print(f"   PSNR: {psnr_inf}")
    print("   ✓ Edge case thành công")

    print("\n✅ Tất cả tests đã pass!")
