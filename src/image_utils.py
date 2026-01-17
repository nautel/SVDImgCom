"""
Module: image_utils.py
Mô tả: Các hàm tiện ích để load, lưu, và xử lý ảnh.

Author: Student - KHTN University
Date: 2026
"""

import numpy as np
import cv2
from PIL import Image
import os


def load_image(filepath, mode='RGB'):
    """
    Load ảnh từ file.

    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file ảnh
    mode : str, optional
        Chế độ load: 'RGB' (màu), 'GRAY' (xám), hoặc 'UNCHANGED' (giữ nguyên)
        Default: 'RGB'

    Returns:
    --------
    numpy.ndarray
        Mảng numpy chứa dữ liệu ảnh
        - Nếu mode='RGB': shape (height, width, 3)
        - Nếu mode='GRAY': shape (height, width)

    Raises:
    -------
    FileNotFoundError
        Nếu file không tồn tại
    ValueError
        Nếu không thể đọc được ảnh

    Examples:
    ---------
    >>> img_rgb = load_image('lena.png', mode='RGB')
    >>> img_gray = load_image('cameraman.jpg', mode='GRAY')
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    try:
        if mode == 'RGB':
            # Load ảnh màu và convert từ BGR (OpenCV) sang RGB
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {filepath}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif mode == 'GRAY':
            # Load ảnh xám trực tiếp
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {filepath}")

        elif mode == 'UNCHANGED':
            # Load ảnh giữ nguyên định dạng
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {filepath}")

        else:
            raise ValueError(f"Mode không hợp lệ: {mode}. Chọn 'RGB', 'GRAY', hoặc 'UNCHANGED'.")

        return image

    except Exception as e:
        raise ValueError(f"Lỗi khi đọc ảnh {filepath}: {str(e)}")


def convert_to_grayscale(image):
    """
    Chuyển đổi ảnh màu sang ảnh xám.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh màu RGB với shape (height, width, 3)

    Returns:
    --------
    numpy.ndarray
        Ảnh xám với shape (height, width)

    Notes:
    ------
    Sử dụng công thức chuẩn: Gray = 0.299*R + 0.587*G + 0.114*B

    Examples:
    ---------
    >>> img_rgb = load_image('peppers.png', mode='RGB')
    >>> img_gray = convert_to_grayscale(img_rgb)
    """
    if len(image.shape) == 2:
        # Ảnh đã là grayscale rồi
        return image

    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Chuyển đổi RGB sang grayscale
        # cv2.cvtColor expects BGR, but we have RGB, so we use appropriate weights
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    else:
        raise ValueError(f"Shape ảnh không hợp lệ: {image.shape}. Cần (H, W) hoặc (H, W, 3).")


def save_image(image, filepath, ensure_uint8=True):
    """
    Lưu ảnh ra file.

    Parameters:
    -----------
    image : numpy.ndarray
        Dữ liệu ảnh cần lưu
    filepath : str
        Đường dẫn file đầu ra
    ensure_uint8 : bool, optional
        Nếu True, tự động convert sang uint8 [0, 255]
        Default: True

    Returns:
    --------
    bool
        True nếu lưu thành công

    Examples:
    ---------
    >>> compressed_img = compress_grayscale(original, k=50)
    >>> save_image(compressed_img, 'results/compressed_k50.png')
    """
    try:
        # Tạo thư mục nếu chưa tồn tại
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Chuẩn bị ảnh để lưu
        img_to_save = image.copy()

        if ensure_uint8:
            # Đảm bảo giá trị trong khoảng [0, 255] và type uint8
            if img_to_save.dtype != np.uint8:
                img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)

        # Nếu là ảnh RGB, convert sang BGR cho OpenCV
        if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

        # Lưu ảnh
        success = cv2.imwrite(filepath, img_to_save)

        if not success:
            raise IOError(f"Không thể lưu ảnh vào {filepath}")

        return True

    except Exception as e:
        raise IOError(f"Lỗi khi lưu ảnh {filepath}: {str(e)}")


def normalize_image(image, target_range='0-255'):
    """
    Normalize giá trị pixel của ảnh.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh cần normalize
    target_range : str, optional
        Khoảng giá trị đích: '0-1' hoặc '0-255'
        Default: '0-255'

    Returns:
    --------
    numpy.ndarray
        Ảnh đã được normalize

    Notes:
    ------
    - '0-1': Normalize về khoảng [0.0, 1.0] (float)
    - '0-255': Normalize về khoảng [0, 255] (uint8)

    Examples:
    ---------
    >>> img_float = normalize_image(img, target_range='0-1')  # [0, 1]
    >>> img_uint8 = normalize_image(img, target_range='0-255')  # [0, 255]
    """
    img = image.copy().astype(np.float64)

    # Normalize về [0, 1]
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min > 0:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)

    if target_range == '0-1':
        return img_normalized

    elif target_range == '0-255':
        img_normalized = (img_normalized * 255).astype(np.uint8)
        return img_normalized

    else:
        raise ValueError(f"target_range không hợp lệ: {target_range}. Chọn '0-1' hoặc '0-255'.")


def get_image_info(image):
    """
    Lấy thông tin về ảnh.

    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh cần lấy thông tin

    Returns:
    --------
    dict
        Dictionary chứa thông tin ảnh (shape, dtype, min, max, mean)

    Examples:
    ---------
    >>> info = get_image_info(img)
    >>> print(f"Shape: {info['shape']}, Type: {info['dtype']}")
    """
    return {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': image.min(),
        'max': image.max(),
        'mean': image.mean(),
        'size_bytes': image.nbytes
    }


if __name__ == "__main__":
    """
    Test các functions trong module.
    """
    print("Testing image_utils.py...")

    # Test với ảnh mẫu (tạo ảnh random để test)
    print("\n1. Tạo ảnh test:")
    test_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    test_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    print("   - Ảnh xám:", test_gray.shape)
    print("   - Ảnh màu:", test_rgb.shape)

    # Test save và load
    print("\n2. Test save_image và load_image:")
    save_image(test_gray, 'test_gray.png')
    save_image(test_rgb, 'test_rgb.png')
    print("   ✓ Đã lưu ảnh test")

    loaded_gray = load_image('test_gray.png', mode='GRAY')
    loaded_rgb = load_image('test_rgb.png', mode='RGB')
    print("   ✓ Đã load ảnh test")

    # Test convert_to_grayscale
    print("\n3. Test convert_to_grayscale:")
    gray_from_rgb = convert_to_grayscale(test_rgb)
    print(f"   RGB {test_rgb.shape} -> Gray {gray_from_rgb.shape}")
    print("   ✓ Chuyển đổi thành công")

    # Test normalize
    print("\n4. Test normalize_image:")
    norm_01 = normalize_image(test_gray, target_range='0-1')
    norm_255 = normalize_image(test_gray, target_range='0-255')
    print(f"   Normalize [0-1]: min={norm_01.min():.3f}, max={norm_01.max():.3f}")
    print(f"   Normalize [0-255]: min={norm_255.min()}, max={norm_255.max()}")
    print("   ✓ Normalize thành công")

    # Test get_image_info
    print("\n5. Test get_image_info:")
    info = get_image_info(test_rgb)
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n✅ Tất cả tests đã pass!")
