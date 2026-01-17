# Nén ảnh bằng phương pháp SVD (Singular Value Decomposition)

**Đề tài môn học:** Phân tích xử lý ảnh
**Sinh viên:** Năm 3 - Đại học Khoa học Tự nhiên (KHTN)
**Ngôn ngữ:** Python
**Phương pháp:** SVD (Singular Value Decomposition)
**Trạng thái:** ✅ **HOÀN THÀNH** (All 3 Phases Complete)

---

## 🎉 Project Status

✅ **Phase 1 COMPLETE** - Core algorithms và modules
✅ **Phase 2 COMPLETE** - Visualizations và analysis
✅ **Phase 3 COMPLETE** - Report outline và deliverables

**Tổng kết:**
- 4 Python modules
- 5 Jupyter notebooks
- 2 test/export scripts
- 30+ visualizations
- 9 high-resolution figures cho báo cáo
- Comprehensive report outline
- Full documentation

**Ready for submission! 📚**

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Lý thuyết SVD](#lý-thuyết-svd)
3. [Cài đặt](#cài-đặt)
4. [Cấu trúc dự án](#cấu-trúc-dự-án)
5. [Sử dụng](#sử-dụng)
6. [Kết quả](#kết-quả)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## 🎯 Giới thiệu

Dự án này triển khai thuật toán **nén ảnh sử dụng SVD (Singular Value Decomposition)**. SVD là một phương pháp trong đại số tuyến tính cho phép phân tích ma trận thành 3 ma trận nhỏ hơn, từ đó có thể nén ảnh bằng cách chỉ giữ lại các thành phần quan trọng nhất.

### Ưu điểm của SVD compression:
- ✅ Dựa trên nền tảng toán học vững chắc
- ✅ Có thể điều chỉnh mức độ nén (thông qua tham số k)
- ✅ Giữ được các đặc trưng quan trọng của ảnh
- ✅ Đơn giản để implement và hiểu

### Hạn chế:
- ❌ Không hiệu quả bằng JPEG cho ảnh tự nhiên
- ❌ Phức tạp tính toán cao (O(mn²) cho ảnh m×n)
- ❌ Cần lưu trữ cả ma trận U, Σ, V^T

---

## 📚 Lý thuyết SVD

### Định nghĩa

Với ma trận ảnh **A** kích thước m×n, SVD phân tích:

```
A = U × Σ × V^T
```

Trong đó:
- **U**: Ma trận m×m (left singular vectors, trực giao)
- **Σ**: Ma trận m×n đường chéo (singular values, giảm dần)
- **V^T**: Ma trận n×n (right singular vectors, trực giao)

### Nén ảnh với SVD

Để nén, ta chỉ giữ **k** singular values lớn nhất:

```
A_compressed = U[:, :k] × Σ[:k, :k] × V^T[:k, :]
```

**Dung lượng:**
- Gốc: `m × n` giá trị
- Nén: `k(m + n + 1)` giá trị
- Tỷ lệ nén: `(1 - k(m+n+1)/(m×n)) × 100%`

**Ví dụ:** Với ảnh 512×512 và k=50:
- Gốc: 262,144 giá trị
- Nén: 51,250 giá trị
- Tiết kiệm: ~80.5%

---

## 🛠️ Cài đặt

### 1. Clone dự án

```bash
cd C:\Users\nuate\project
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Thư viện cần thiết:
- `numpy` - Tính toán ma trận và SVD
- `matplotlib` - Vẽ biểu đồ
- `opencv-python` - Xử lý ảnh
- `Pillow` - Load/save ảnh
- `scikit-image` - Metrics (SSIM)
- `jupyter` - Chạy notebooks
- `pandas` - Xử lý dữ liệu
- `seaborn` - Visualizations

---

## 📁 Cấu trúc dự án

```
project/
│
├── README.md                          # File này - hướng dẫn sử dụng
├── requirements.txt                   # Dependencies
│
├── src/                               # Source code
│   ├── svd_compression.py            # Thuật toán nén SVD (CORE)
│   ├── image_utils.py                # Load/save ảnh
│   ├── quality_metrics.py            # Tính PSNR, MSE, SSIM
│   └── visualization.py              # Vẽ biểu đồ và phân tích
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_svd_theory_demo.ipynb      # Demo lý thuyết SVD
│   ├── 02_grayscale_compression.ipynb # Nén ảnh xám
│   ├── 03_color_compression.ipynb     # Nén ảnh màu
│   ├── 04_comparative_analysis.ipynb  # So sánh grayscale vs RGB
│   └── 05_final_summary.ipynb         # Tổng kết toàn bộ dự án
│
├── images/                            # Ảnh test
│   ├── grayscale/                    # Ảnh trắng đen
│   └── color/                        # Ảnh màu
│
├── results/                           # Kết quả
│   ├── compressed/                   # Ảnh đã nén
│   ├── visualizations/               # Biểu đồ
│   └── metrics/                      # Bảng metrics
│
├── report/                            # Báo cáo
│   ├── report_outline.md             # Outline báo cáo chi tiết (8 phần)
│   └── figures/                      # Hình ảnh cho báo cáo (9 figures)
│
├── test_user_images.py               # Script test với ảnh thực
└── export_report_figures.py          # Script export figures cho báo cáo
```

---

## 🚀 Sử dụng

### Option 1: Sử dụng Jupyter Notebooks (Khuyến nghị)

Notebooks cung cấp demo chi tiết với visualization:

```bash
jupyter notebook
```

Mở các notebooks theo thứ tự:
1. **01_svd_theory_demo.ipynb** - Hiểu lý thuyết SVD với demo đơn giản
2. **02_grayscale_compression.ipynb** - Nén ảnh xám (5.3.01.tiff)
3. **03_color_compression.ipynb** - Nén ảnh RGB (4.1.01.tiff)
4. **04_comparative_analysis.ipynb** - So sánh grayscale vs RGB
5. **05_final_summary.ipynb** - Tổng kết toàn bộ dự án

### Option 2: Sử dụng Python code trực tiếp

```python
from src.image_utils import load_image, save_image
from src.svd_compression import compress_grayscale, compress_rgb
from src.quality_metrics import calculate_psnr, calculate_mse

# Load ảnh
original = load_image('images/grayscale/lena.png', mode='GRAY')

# Nén với k=50
compressed = compress_grayscale(original, k=50)

# Đánh giá chất lượng
psnr = calculate_psnr(original, compressed)
mse = calculate_mse(original, compressed)

print(f"PSNR: {psnr:.2f} dB")
print(f"MSE: {mse:.2f}")

# Lưu ảnh
save_image(compressed, 'results/compressed/output_k50.png')
```

### Option 3: Chạy Test Scripts

**Test với ảnh thực:**
```bash
python test_user_images.py
```
Script này sẽ:
- Test compression trên cả grayscale và RGB
- Tạo tất cả visualizations
- Save kết quả vào results/

**Export figures cho báo cáo:**
```bash
python export_report_figures.py
```
Script này sẽ:
- Tạo 9 figures chất lượng cao (DPI 300)
- Save vào report/figures/
- Sẵn sàng chèn vào báo cáo

---

## 📊 API Reference

### Module: `svd_compression.py`

#### `compress_grayscale(image, k)`
Nén ảnh xám sử dụng SVD.

**Parameters:**
- `image`: numpy array (m×n), ảnh xám
- `k`: int, số singular values giữ lại

**Returns:**
- numpy array (m×n), ảnh đã nén

**Example:**
```python
compressed = compress_grayscale(original, k=50)
```

#### `compress_rgb(image, k)`
Nén ảnh màu RGB (áp dụng SVD cho từng kênh).

**Parameters:**
- `image`: numpy array (m×n×3), ảnh RGB
- `k`: int, số singular values cho mỗi kênh

**Returns:**
- numpy array (m×n×3), ảnh đã nén

#### `get_svd_matrices(image)`
Lấy các ma trận U, Σ, V^T từ SVD.

**Returns:**
- tuple (U, S, Vt)

#### `calculate_compression_ratio(original_shape, k, is_rgb=False)`
Tính tỷ lệ nén.

**Returns:**
- dict với keys: `original_size`, `compressed_size`, `space_saved_percent`

### Module: `quality_metrics.py`

#### `calculate_mse(original, compressed)`
Tính Mean Squared Error.

**Formula:**
```
MSE = (1/(m×n)) × Σ(original - compressed)²
```

#### `calculate_psnr(original, compressed, max_pixel_value=255)`
Tính Peak Signal-to-Noise Ratio.

**Formula:**
```
PSNR = 10 × log₁₀(255²/MSE) dB
```

**Interpretation:**
- PSNR > 40 dB: Excellent
- PSNR 30-40 dB: Good
- PSNR 20-30 dB: Fair
- PSNR < 20 dB: Poor

#### `calculate_all_metrics(original, compressed, include_ssim=False)`
Tính tất cả metrics cùng lúc.

**Returns:**
- dict với keys: `mse`, `rmse`, `mae`, `psnr`, `ssim` (optional)

---

## 📈 Kết quả

### Kết quả thực tế trên ảnh test (512×512)

**Grayscale Image (5.3.01.tiff):**

| k   | PSNR (dB) | SSIM   | Space Saved | Chất lượng |
|-----|-----------|--------|-------------|------------|
| 5   | ~22       | ~0.65  | 98.0%       | Poor       |
| 10  | ~28       | ~0.85  | 96.1%       | Fair       |
| 20  | ~34       | ~0.94  | 92.2%       | Good       |
| 50  | ~41       | ~0.98  | 80.5%       | Excellent  |
| 100 | ~47       | ~0.99  | 61.9%       | Excellent  |

**RGB Image (4.1.01.tiff):**

| k   | PSNR (dB) | Space Saved | Chất lượng |
|-----|-----------|-------------|------------|
| 5   | ~20       | 98.0%       | Poor       |
| 10  | ~26       | 96.1%       | Fair       |
| 20  | ~32       | 92.2%       | Good       |
| 50  | ~39       | 80.5%       | Good       |
| 100 | ~45       | 61.9%       | Excellent  |

### Key Findings:

1. **Optimal k**:
   - k=20-30 cho balance tốt (PSNR ≥ 30dB, Space saved >90%)
   - k=50 cho quality cao (PSNR ≥ 40dB)

2. **Grayscale vs RGB**:
   - Compression ratio giống nhau cho cùng k
   - RGB cần xử lý 3× data nhưng giữ được màu sắc

3. **Trade-off rõ ràng**:
   - k nhỏ: nén mạnh, chất lượng thấp
   - k lớn: chất lượng cao, nén ít

4. **Cumulative energy**:
   - 90% năng lượng ở k ≈ 30-40
   - 95% năng lượng ở k ≈ 50-60

---

## 🧪 Testing

Chạy tests cho các modules:

```bash
# Test image_utils
python src/image_utils.py

# Test svd_compression
python src/svd_compression.py

# Test quality_metrics
python src/quality_metrics.py
```

Tất cả modules đều có `if __name__ == "__main__"` block để self-test.

---

## 📖 Tài liệu tham khảo

### Videos (Khuyến nghị xem):
1. **Steve Brunton - SVD Playlist**
   https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv

2. **3Blue1Brown - Linear Algebra Series**
   https://www.youtube.com/c/3blue1brown

3. **Computerphile - Image Compression**
   https://www.youtube.com/watch?v=Q2aEzeMDHMA

### Sách:
1. Gilbert Strang - "Introduction to Linear Algebra" (Chapter 7: SVD)
2. Gonzalez & Woods - "Digital Image Processing" (Chapter 8)
3. "Tutorial on SVD" by Kirk Baker

### Documentation:
- [NumPy SVD Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [scikit-image Metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

### Papers:
- Eckart, C., & Young, G. (1936). "The approximation of one matrix by another of lower rank"
- Andrews, H. C., & Patterson III, C. L. (1976). "Singular value decomposition (SVD) image coding"

---

## 🎓 Học tập

### Để hiểu sâu hơn:

1. **Lý thuyết Linear Algebra**:
   - Eigenvalues và eigenvectors
   - Orthogonal matrices
   - Matrix decomposition

2. **Image Processing**:
   - Color spaces (RGB, YCbCr)
   - Quality metrics (PSNR, SSIM)
   - Compression techniques

3. **Thực hành**:
   - Chạy notebooks từng bước
   - Thử với ảnh của riêng bạn
   - Thay đổi k và quan sát kết quả
   - So sánh với JPEG compression

---

## 🔧 Troubleshooting

### Lỗi thường gặp:

**1. ImportError: No module named 'cv2'**
```bash
pip install opencv-python
```

**2. Shape mismatch error**
- Đảm bảo ảnh grayscale có shape (m, n), không phải (m, n, 1)
- Dùng `convert_to_grayscale()` để convert

**3. PSNR = inf**
- Xảy ra khi MSE = 0 (ảnh gốc = ảnh nén)
- Normal nếu k = min(m, n)

**4. Memory error với ảnh lớn**
- Giảm kích thước ảnh trước khi xử lý
- Dùng `full_matrices=False` trong `np.linalg.svd()`

---

## ✅ Hoàn thành (Completed Deliverables)

### ✅ Phase 1 - Cơ sở và thuật toán (HOÀN THÀNH)
- ✅ 4 modules Python (image_utils, svd_compression, quality_metrics, visualization)
- ✅ Notebook 01: SVD Theory Demo
- ✅ Notebook 02: Grayscale Compression
- ✅ Full API với docstrings
- ✅ Testing functions

### ✅ Phase 2 - Visualization và phân tích (HOÀN THÀNH)
- ✅ visualization.py module (7 plot functions)
- ✅ Notebook 03: RGB compression
- ✅ Notebook 04: Comparative analysis
- ✅ Notebook 05: Final summary
- ✅ test_user_images.py script
- ✅ export_report_figures.py script
- ✅ 30+ visualizations created

### ✅ Phase 3 - Báo cáo (HOÀN THÀNH)
- ✅ report_outline.md (8 sections, comprehensive)
- ✅ 9 high-resolution figures (DPI 300)
- ✅ Final summary notebook
- ✅ CSV/JSON data exports
- ✅ Comprehensive README

### 🚀 Nâng cao (Future Work)
- [ ] CLI tool với argparse
- [ ] Block-based SVD compression (8×8 blocks như JPEG)
- [ ] Adaptive k selection algorithm
- [ ] GUI với Streamlit/Gradio
- [ ] Benchmark với JPEG compression
- [ ] YCbCr color space
- [ ] GPU acceleration (CuPy/PyTorch)

---

## 👨‍🎓 Tác giả

**Sinh viên năm 3**
Đại học Khoa học Tự nhiên (KHTN)
Môn học: Phân tích xử lý ảnh
Năm: 2026

---

## 📄 License

Dự án này được tạo cho mục đích học tập. Free to use for educational purposes.

---

## 🙏 Lời cảm ơn

- Giảng viên môn Phân tích xử lý ảnh
- Tài liệu từ MIT OpenCourseWare
- Steve Brunton's YouTube channel
- NumPy và OpenCV communities

---

**Happy Learning! 📚✨**

Nếu có câu hỏi, hãy tham khảo notebooks hoặc tài liệu tham khảo ở trên.
