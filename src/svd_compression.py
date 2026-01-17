"""
Module: svd_compression.py
Mô tả: Thuật toán nén ảnh sử dụng Singular Value Decomposition (SVD).

Lý thuyết SVD:
--------------
SVD phân tích ma trận A (ảnh) thành 3 ma trận:
    A = U × Σ × V^T

Trong đó:
- U: Ma trận singular vectors trái (m × m, orthogonal)
- Σ: Ma trận đường chéo chứa singular values (m × n, giảm dần)
- V^T: Ma trận singular vectors phải (n × n, orthogonal)

Để nén ảnh, ta chỉ giữ lại k singular values lớn nhất:
    A_compressed = U[:, :k] × Σ[:k, :k] × V^T[:k, :]

Điều này giảm dung lượng từ m×n xuống k(m+n+1) giá trị.

Author: Student - KHTN University
Date: 2026
"""

import numpy as np
from typing import Tuple, Optional


def compress_grayscale(image, k):
    """
    Nén ảnh xám sử dụng SVD với k singular values.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh xám đầu vào, shape (m, n)
    k : int
        Số lượng singular values giữ lại (k <= min(m, n))

    Returns:
    --------
    numpy.ndarray
        Ảnh đã nén, shape (m, n), dtype uint8

    Algorithm:
    ----------
    1. Convert ảnh sang float64 để tính toán chính xác
    2. Áp dụng SVD: U, S, Vt = np.linalg.svd(image)
    3. Giữ k thành phần: U[:, :k], S[:k], Vt[:k, :]
    4. Tái tạo: compressed = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    5. Clip giá trị về [0, 255] và convert sang uint8

    Raises:
    -------
    ValueError
        Nếu k không hợp lệ hoặc ảnh không phải 2D

    Examples:
    ---------
    >>> import cv2
    >>> img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    >>> compressed = compress_grayscale(img, k=50)
    >>> print(f"Original: {img.shape}, Compressed with k={50}")
    """
    # Kiểm tra input
    if len(image.shape) != 2:
        raise ValueError(f"Ảnh xám phải có 2 chiều, nhận được: {image.shape}")

    m, n = image.shape
    max_k = min(m, n)

    if k < 1 or k > max_k:
        raise ValueError(f"k phải trong khoảng [1, {max_k}], nhận được k={k}")

    # Bước 1: Convert sang float64 để tính toán
    image_float = image.astype(np.float64)

    # Bước 2: Áp dụng SVD
    # full_matrices=False để tiết kiệm bộ nhớ (chỉ tính min(m,n) components)
    U, S, Vt = np.linalg.svd(image_float, full_matrices=False)

    # Giải thích kết quả SVD:
    # U: (m, min(m,n)) - Left singular vectors
    # S: (min(m,n),) - Singular values (đã sắp xếp giảm dần)
    # Vt: (min(m,n), n) - Right singular vectors (đã transpose)

    # Bước 3: Giữ k thành phần lớn nhất
    U_k = U[:, :k]  # (m, k)
    S_k = S[:k]     # (k,)
    Vt_k = Vt[:k, :]  # (k, n)

    # Bước 4: Tái tạo ảnh
    # Cách 1: Sử dụng @ operator (matrix multiplication)
    compressed = U_k @ np.diag(S_k) @ Vt_k

    # Cách 2 (tối ưu hơn, tránh tạo ma trận đường chéo lớn):
    # compressed = (U_k * S_k) @ Vt_k

    # Bước 5: Clip về [0, 255] và convert sang uint8
    # Lý do clip: do sai số số học, giá trị có thể < 0 hoặc > 255
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    return compressed


def compress_rgb(image, k):
    """
    Nén ảnh màu RGB bằng cách áp dụng SVD cho từng kênh riêng biệt.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh màu RGB, shape (m, n, 3)
    k : int
        Số lượng singular values giữ lại cho mỗi kênh

    Returns:
    --------
    numpy.ndarray
        Ảnh đã nén, shape (m, n, 3), dtype uint8

    Algorithm:
    ----------
    1. Tách ảnh thành 3 kênh: R, G, B
    2. Áp dụng compress_grayscale() cho từng kênh độc lập
    3. Ghép 3 kênh lại thành ảnh RGB

    Notes:
    ------
    - Mỗi kênh được nén độc lập với cùng k
    - Có thể dùng k khác nhau cho mỗi kênh (advanced)
    - Kênh Green thường cần k lớn hơn vì mắt người nhạy cảm hơn

    Examples:
    ---------
    >>> img_rgb = cv2.imread('peppers.png')
    >>> img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    >>> compressed = compress_rgb(img_rgb, k=50)
    """
    # Kiểm tra input
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Ảnh RGB phải có shape (m, n, 3), nhận được: {image.shape}")

    # Bước 1: Tách thành 3 kênh
    R = image[:, :, 0]  # Red channel
    G = image[:, :, 1]  # Green channel
    B = image[:, :, 2]  # Blue channel

    # Bước 2: Nén từng kênh riêng biệt
    R_compressed = compress_grayscale(R, k)
    G_compressed = compress_grayscale(G, k)
    B_compressed = compress_grayscale(B, k)

    # Bước 3: Ghép lại thành ảnh RGB
    compressed = np.stack([R_compressed, G_compressed, B_compressed], axis=2)

    return compressed


def get_svd_matrices(image):
    """
    Lấy các ma trận U, Σ, V^T từ SVD để visualize.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh xám, shape (m, n)

    Returns:
    --------
    tuple
        (U, S, Vt) - Ba ma trận từ SVD

    Notes:
    ------
    Hàm này hữu ích để:
    - Visualize các ma trận U, Σ, V^T
    - Phân tích singular values
    - Hiểu cấu trúc của SVD

    Examples:
    ---------
    >>> U, S, Vt = get_svd_matrices(image)
    >>> plt.semilogy(S)  # Plot singular values
    >>> plt.title('Singular Value Spectrum')
    """
    if len(image.shape) != 2:
        raise ValueError(f"Chỉ hỗ trợ ảnh xám 2D, nhận được: {image.shape}")

    # Convert sang float64
    image_float = image.astype(np.float64)

    # Áp dụng SVD
    U, S, Vt = np.linalg.svd(image_float, full_matrices=False)

    return U, S, Vt


def reconstruct_from_svd(U, S, Vt, k):
    """
    Tái tạo ảnh từ các ma trận SVD với k components.

    Parameters:
    -----------
    U : numpy.ndarray
        Left singular vectors, shape (m, min(m,n))
    S : numpy.ndarray
        Singular values, shape (min(m,n),)
    Vt : numpy.ndarray
        Right singular vectors (transposed), shape (min(m,n), n)
    k : int
        Số lượng components giữ lại

    Returns:
    --------
    numpy.ndarray
        Ảnh tái tạo, shape (m, n), dtype uint8

    Examples:
    ---------
    >>> U, S, Vt = get_svd_matrices(image)
    >>> reconstructed_k10 = reconstruct_from_svd(U, S, Vt, k=10)
    >>> reconstructed_k50 = reconstruct_from_svd(U, S, Vt, k=50)
    """
    max_k = len(S)

    if k < 1 or k > max_k:
        raise ValueError(f"k phải trong khoảng [1, {max_k}], nhận được k={k}")

    # Giữ k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Tái tạo
    # Sử dụng (U_k * S_k) thay vì U_k @ diag(S_k) để tối ưu
    reconstructed = (U_k * S_k) @ Vt_k

    # Clip và convert
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return reconstructed


def calculate_compression_ratio(original_shape, k, is_rgb=False):
    """
    Tính tỷ lệ nén (compression ratio) cho SVD.

    Parameters:
    -----------
    original_shape : tuple
        Shape của ảnh gốc (m, n) hoặc (m, n, 3)
    k : int
        Số singular values giữ lại
    is_rgb : bool, optional
        True nếu là ảnh màu RGB
        Default: False

    Returns:
    --------
    dict
        Dictionary chứa:
        - original_size: Dung lượng gốc (số phần tử)
        - compressed_size: Dung lượng nén (số phần tử)
        - ratio_percent: Tỷ lệ nén (%)
        - space_saved_percent: Phần trăm tiết kiệm dung lượng (%)

    Formula:
    --------
    Grayscale:
        Original: m × n
        Compressed: k(m + n + 1)  [U: m×k, S: k, Vt: k×n]

    RGB:
        Original: m × n × 3
        Compressed: 3 × k(m + n + 1)

    Compression ratio = (1 - compressed/original) × 100%

    Examples:
    ---------
    >>> stats = calculate_compression_ratio((512, 512), k=50, is_rgb=False)
    >>> print(f"Tiết kiệm: {stats['space_saved_percent']:.1f}%")
    """
    if is_rgb:
        m, n, channels = original_shape
        if channels != 3:
            raise ValueError(f"RGB image phải có 3 kênh, nhận được {channels}")

        # Original size
        original_size = m * n * 3

        # Compressed size: 3 kênh × [k×m + k + k×n]
        compressed_size = 3 * k * (m + n + 1)

    else:
        m, n = original_shape

        # Original size
        original_size = m * n

        # Compressed size: k×m (U) + k (S) + k×n (Vt)
        compressed_size = k * (m + n + 1)

    # Tính tỷ lệ nén
    ratio = compressed_size / original_size
    ratio_percent = ratio * 100
    space_saved_percent = (1 - ratio) * 100

    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': ratio,
        'ratio_percent': ratio_percent,
        'space_saved_percent': space_saved_percent,
        'k': k
    }


def calculate_cumulative_energy(S, k=None):
    """
    Tính năng lượng tích lũy (cumulative energy) của k singular values.

    Parameters:
    -----------
    S : numpy.ndarray
        Mảng singular values
    k : int, optional
        Số singular values. Nếu None, tính cho tất cả.

    Returns:
    --------
    float hoặc numpy.ndarray
        Phần trăm năng lượng được giữ lại bởi k singular values
        Nếu k=None, trả về array của cumulative energy cho tất cả k

    Formula:
    --------
    Energy(k) = (Σ σᵢ² for i=1 to k) / (Σ σᵢ² for i=1 to n) × 100%

    Notes:
    ------
    Năng lượng đại diện cho "lượng thông tin" trong ảnh.
    Thường 90% năng lượng được giữ lại với k << n.

    Examples:
    ---------
    >>> U, S, Vt = get_svd_matrices(image)
    >>> energy_k50 = calculate_cumulative_energy(S, k=50)
    >>> print(f"Năng lượng với k=50: {energy_k50:.1f}%")
    >>>
    >>> # Tính cho tất cả k
    >>> energies = calculate_cumulative_energy(S)
    >>> plt.plot(energies)
    """
    # Tính tổng bình phương của tất cả singular values
    total_energy = np.sum(S ** 2)

    if k is None:
        # Tính cumulative energy cho tất cả k
        cumulative_energy = np.cumsum(S ** 2) / total_energy * 100
        return cumulative_energy
    else:
        # Tính energy cho k cụ thể
        if k < 1 or k > len(S):
            raise ValueError(f"k phải trong khoảng [1, {len(S)}], nhận được k={k}")

        energy_k = np.sum(S[:k] ** 2) / total_energy * 100
        return energy_k


if __name__ == "__main__":
    """
    Test các functions trong module.
    """
    print("Testing svd_compression.py...")

    # Tạo ảnh test
    print("\n1. Tạo ảnh test:")
    test_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    test_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    print(f"   Ảnh xám: {test_gray.shape}")
    print(f"   Ảnh màu: {test_rgb.shape}")

    # Test compress_grayscale
    print("\n2. Test compress_grayscale:")
    for k in [5, 10, 20, 50]:
        compressed = compress_grayscale(test_gray, k)
        print(f"   k={k:2d}: shape={compressed.shape}, dtype={compressed.dtype}")
    print("   ✓ Nén grayscale thành công")

    # Test compress_rgb
    print("\n3. Test compress_rgb:")
    compressed_rgb = compress_rgb(test_rgb, k=20)
    print(f"   RGB compressed: {compressed_rgb.shape}")
    print("   ✓ Nén RGB thành công")

    # Test get_svd_matrices
    print("\n4. Test get_svd_matrices:")
    U, S, Vt = get_svd_matrices(test_gray)
    print(f"   U shape: {U.shape}")
    print(f"   S shape: {S.shape}")
    print(f"   Vt shape: {Vt.shape}")
    print(f"   Top 5 singular values: {S[:5]}")
    print("   ✓ Get SVD matrices thành công")

    # Test reconstruct_from_svd
    print("\n5. Test reconstruct_from_svd:")
    reconstructed = reconstruct_from_svd(U, S, Vt, k=30)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print("   ✓ Reconstruction thành công")

    # Test calculate_compression_ratio
    print("\n6. Test calculate_compression_ratio:")
    stats_gray = calculate_compression_ratio((100, 100), k=20, is_rgb=False)
    stats_rgb = calculate_compression_ratio((100, 100, 3), k=20, is_rgb=True)

    print(f"   Grayscale (k=20):")
    print(f"      Original: {stats_gray['original_size']} values")
    print(f"      Compressed: {stats_gray['compressed_size']} values")
    print(f"      Space saved: {stats_gray['space_saved_percent']:.1f}%")

    print(f"   RGB (k=20):")
    print(f"      Original: {stats_rgb['original_size']} values")
    print(f"      Compressed: {stats_rgb['compressed_size']} values")
    print(f"      Space saved: {stats_rgb['space_saved_percent']:.1f}%")
    print("   ✓ Compression ratio thành công")

    # Test calculate_cumulative_energy
    print("\n7. Test calculate_cumulative_energy:")
    energy_k10 = calculate_cumulative_energy(S, k=10)
    energy_k50 = calculate_cumulative_energy(S, k=50)
    energies_all = calculate_cumulative_energy(S)

    print(f"   Energy với k=10: {energy_k10:.2f}%")
    print(f"   Energy với k=50: {energy_k50:.2f}%")
    print(f"   Energy với k=100 (max): {energies_all[-1]:.2f}%")
    print("   ✓ Cumulative energy thành công")

    print("\n✅ Tất cả tests đã pass!")
