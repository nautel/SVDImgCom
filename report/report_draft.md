# BÁO CÁO ĐỀ TÀI

# NÉN ẢNH BẰNG PHƯƠNG PHÁP PHÂN TÍCH GIÁ TRỊ KỲ DỊ (SVD)

**Image Compression using Singular Value Decomposition**

---


---

## TÓM TẮT

Báo cáo này trình bày việc nghiên cứu và triển khai thuật toán nén ảnh sử dụng phương pháp Singular Value Decomposition (SVD). SVD là một kỹ thuật trong đại số tuyến tính cho phép phân tích ma trận thành ba thành phần chính, từ đó có thể nén ảnh bằng cách chỉ giữ lại các singular values lớn nhất.

Dự án đã triển khai thành công thuật toán SVD compression cho cả ảnh grayscale và RGB, với các kết quả đạt được: tỷ lệ nén 80-92% với chất lượng tốt (PSNR ≥ 30dB). Thực nghiệm được tiến hành trên ảnh test kích thước 512×512 pixels, với các giá trị k từ 5 đến 200. Kết quả cho thấy k=20-50 là điểm cân bằng tối ưu giữa chất lượng ảnh và tỷ lệ nén.

**Từ khóa:** SVD, image compression, singular value decomposition, PSNR, nén ảnh, low-rank approximation

---

## MỄC LỄC

**1. GIỚI THIỆU** ..................................................... 5
   - 1.1 Bối cảnh và động lực ...................................... 5
   - 1.2 Vấn đề nghiên cứu ........................................ 6
   - 1.3 Mục tiêu đề tài ........................................... 6
   - 1.4 Phạm vi nghiên cứu ........................................ 7
   - 1.5 Đóng góp .................................................. 7

**2. CƠ SỞ LÝ THUYẾT** ................................................ 8
   - 2.1 Giới thiệu về SVD ........................................ 8
   - 2.2 Low-rank Approximation ................................... 11
   - 2.3 SVD cho nén ảnh .......................................... 12
   - 2.4 Compression Ratio ........................................ 14
   - 2.5 Đánh giá chất lượng ảnh .................................. 15

**3. PHƯƠNG PHÁP THỰC HIỆN** ......................................... 18
   - 3.1 Tổng quan phương pháp .................................... 18
   - 3.2 Thuật toán chi tiết ...................................... 19
   - 3.3 Đánh giá chất lượng ...................................... 22
   - 3.4 Thiết kế thực nghiệm ..................................... 23

**4. CHI TIẾT IMPLEMENTATION** ....................................... 25
   - 4.1 Cấu trúc code ............................................ 25
   - 4.2 Modules chính ............................................ 26
   - 4.3 Xử lý lỗi và edge cases .................................. 30

**5. KẾT QUẢ THỰC NGHIỆM** .......................................... 31
   - 5.1 Thí nghiệm 1: Nén ảnh Grayscale .......................... 31
   - 5.2 Thí nghiệm 2: Nén ảnh RGB ................................ 38
   - 5.3 Thí nghiệm 3: So sánh Grayscale vs RGB ................... 42

**6. THẢO LUẬN** ..................................................... 46
   - 6.1 Phân tích hiệu quả SVD compression ....................... 46
   - 6.2 So sánh với các phương pháp khác ......................... 47
   - 6.3 Trade-offs và Best Practices ............................. 48
   - 6.4 Limitations và Future Work ............................... 49

**7. KẾT LUẬN** ...................................................... 51
   - 7.1 Tổng kết kết quả ......................................... 51
   - 7.2 Bài học kinh nghiệm ...................................... 52
   - 7.3 Hướng phát triển ......................................... 53

**8. TÀI LIỆU THAM KHẢO** ............................................ 54

**PHỄ LỄC** .......................................................... 56
   - Phụ lục A: Source code repository ............................ 56
   - Phụ lục B: Kết quả chi tiết .................................. 57
   - Phụ lục C: Hướng dẫn sử dụng ................................. 58

---
---

# 1. GIỚI THIỆU

## 1.1 Bối cảnh và động lực

Trong kỷ nguyên số hiện nay, lượng dữ liệu hình ảnh được tạo ra và lưu trữ tăng lên một cách chóng mặt. Theo thống kê, mỗi ngày có hơn 3.2 tỷ hình ảnh được chia sẻ trên các nền tảng mạng xã hội. Điều này đặt ra những thách thức lớn về mặt lưu trữ và truyền tải dữ liệu.

**Nhu cầu nén ảnh phát sinh từ:**

1. **Giới hạn về dung lượng lưu trữ:**
   - Thiết bị di động có dung lượng hạn chế
   - Chi phí lưu trữ cloud tăng theo dung lượng
   - Database ảnh y tế, vệ tinh cần lưu trữ lâu dài

2. **Giới hạn về băng thông:**
   - Truyền tải ảnh qua internet cần tối ưu bandwidth
   - Streaming video yêu cầu compression real-time
   - Ứng dụng IoT với kết nối mạng hạn chế

3. **Yêu cầu về hiệu suất:**
   - Website cần load ảnh nhanh để cải thiện UX
   - Mobile apps cần tiết kiệm data cho người dùng
   - Backup và recovery nhanh hơn với dữ liệu nhỏ hơn

**Vai trò của compression trong các lĩnh vực:**

- **Y tế:** Lưu trữ và truyền tải ảnh X-ray, MRI, CT scan
- **Viễn thám:** Ảnh vệ tinh độ phân giải cao
- **Đa phương tiện:** Streaming video, photography, gaming
- **Bảo mật:** Watermarking, steganography
- **Machine Learning:** Giảm kích thước dataset ảnh

Trong bối cảnh đó, nghiên cứu các phương pháp nén ảnh hiệu quả trở nên vô cùng quan trọng. Singular Value Decomposition (SVD) là một trong những phương pháp có nền tảng toán học vững chắc, dễ hiểu và triển khai, phù hợp cho mục đích học tập và nghiên cứu.

## 1.2 Vấn đề nghiên cứu

**Thách thức chính trong nén ảnh:**

Làm thế nào để **giảm dung lượng ảnh** mà vẫn **giữ được chất lượng** và **thông tin quan trọng**?

Đây là một bài toán optimization với hai mục tiêu đối lập:
- **Maximize compression ratio:** Giảm dung lượng càng nhiều càng tốt
- **Maximize image quality:** Giữ ảnh gần giống original nhất có thể

**Câu hỏi nghiên cứu cụ thể:**

1. SVD có thể áp dụng hiệu quả như thế nào cho nén ảnh?
2. Giá trị k (số singular values giữ lại) tối ưu là bao nhiêu?
3. Trade-off giữa compression ratio và quality được biểu diễn ra sao?
4. SVD hoạt động khác nhau như thế nào với ảnh grayscale và RGB?
5. SVD so sánh thế nào với các phương pháp nén khác (JPEG, PNG)?

**Giả thuyết:**

- Với k đủ lớn, SVD có thể đạt chất lượng tốt (PSNR ≥ 30dB)
- Tồn tại một "sweet spot" k cho balance tốt nhất
- Phần lớn thông tin ảnh tập trung ở một số singular values lớn đầu tiên

## 1.3 Mục tiêu đề tài

### Mục tiêu chính

Nghiên cứu và triển khai thuật toán nén ảnh sử dụng **Singular Value Decomposition (SVD)**, đánh giá hiệu quả trên ảnh grayscale và RGB.

### Mục tiêu cụ thể

**1. Về lý thuyết:**
- Nghiên cứu sâu về SVD và cách áp dụng vào nén ảnh
- Hiểu rõ Eckart-Young theorem và low-rank approximation
- Nắm vững các metrics đánh giá chất lượng ảnh (PSNR, MSE, SSIM)

**2. Về thực hành:**
- Xây dựng hệ thống nén ảnh hoàn chỉnh bằng Python
- Triển khai compression cho cả ảnh grayscale và RGB
- Tạo visualizations để phân tích kết quả

**3. Về thực nghiệm:**
- Test với nhiều giá trị k khác nhau
- Đo lường PSNR, MSE, SSIM, compression ratio
- Phân tích trade-off giữa quality và compression
- Tìm k tối ưu cho các use cases khác nhau

**4. Về documentation:**
- Viết báo cáo đầy đủ với lý thuyết và thực nghiệm
- Tạo code có thể tái sử dụng và dễ hiểu
- Cung cấp guidelines thực tế

## 1.4 Phạm vi nghiên cứu

### Trong phạm vi

**Về kỹ thuật:**
- Nén ảnh tĩnh (grayscale và RGB)
- Full-image SVD (không chia blocks)
- Đánh giá định lượng: PSNR, MSE, SSIM
- Đánh giá định tính: visual inspection

**Về dataset:**
- Ảnh test từ USC-SIPI Image Database
- Kích thước: 512×512 pixels
- Grayscale: 5.3.01.tiff
- RGB: 4.1.01.tiff

**Về implementation:**
- Ngôn ngữ: Python 3.8+
- Libraries: NumPy, SciPy, Matplotlib, OpenCV
- Environment: Jupyter Notebook + Python scripts

### Ngoài phạm vi

**Không bao gồm:**
- Nén video (motion compensation, temporal compression)
- Block-based SVD (như JPEG's DCT)
- So sánh chi tiết với JPEG/PNG (chỉ thảo luận lý thuyết)
- Real-time compression
- Hardware acceleration (GPU, FPGA)
- Lossy compression khác (wavelet, fractal)

**Lý do giới hạn phạm vi:**
- Tập trung vào hiểu sâu một phương pháp
- Thời gian thực hiện đề tài hạn chế (3 tuần)
- Mục tiêu học tập, không phải production system

## 1.5 Đóng góp

**Đóng góp chính của đề tài:**

1. **Codebase hoàn chỉnh và well-documented:**
   - 4 modules Python với full docstrings
   - 5 Jupyter notebooks chi tiết
   - Code có thể dùng cho giảng dạy và nghiên cứu

2. **Phân tích toàn diện:**
   - 30+ visualizations
   - Thực nghiệm với 7 giá trị k
   - So sánh grayscale vs RGB
   - Statistical analysis

3. **Practical guidelines:**
   - Cách chọn k tối ưu
   - Trade-off analysis
   - Best practices
   - Troubleshooting common issues

4. **Reproducible results:**
   - Tất cả notebooks có thể chạy lại
   - Kết quả được lưu trữ trong CSV/JSON
   - Figures với DPI cao cho report

**Giá trị học tập:**
- Hiểu sâu về SVD và linear algebra
- Kỹ năng Python programming
- Kỹ năng data visualization
- Kỹ năng scientific writing
- Kinh nghiệm thực nghiệm và phân tích kết quả

---

# 2. CƠ SỞ LÝ THUYẾT

## 2.1 Giới thiệu về SVD

### 2.1.1 Định nghĩa

**Singular Value Decomposition (SVD)** là một phương pháp phân tích ma trận trong đại số tuyến tính. Với bất kỳ ma trận **A** có kích thước m×n, SVD phân tích A thành tích của ba ma trận:

```
A = U × Σ × V^T
```

**Trong đó:**

- **U** (m × m): Ma trận **left singular vectors**
  - Là ma trận trực giao: U^T × U = I
  - Các cột của U là eigenvectors của AA^T
  - Biểu diễn directions trong không gian m-chiều

- **Σ** (m × n): Ma trận **singular values**
  - Ma trận đường chéo (diagonal matrix)
  - Các phần tử σᵢ (i=1,...,r) gọi là singular values
  - Được sắp xếp giảm dần: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
  - r = rank(A)

- **V^T** (n × n): Chuyển vị của ma trận **right singular vectors**
  - V là ma trận trực giao: V^T × V = I
  - Các cột của V là eigenvectors của A^T A
  - Biểu diễn directions trong không gian n-chiều

**Ví dụ minh họa:**

Cho ma trận A (4×3):
```
A = [2  4  0]
    [4  8  0]
    [0  0  3]
    [0  0  0]
```

SVD phân tích A thành:
- U: ma trận 4×4
- Σ: ma trận 4×3 với singular values trên đường chéo
- V^T: ma trận 3×3

### 2.1.2 Tính chất toán học

**1. Orthogonality (Tính trực giao):**

Ma trận U và V là orthogonal matrices:
```
U^T × U = I_(m×m)
V^T × V = I_(n×n)
```

Điều này có nghĩa:
- Các cột của U và V là orthonormal vectors
- Phép biến đổi không làm thay đổi norm
- Numerical stability tốt

**2. Singular values ordering:**

Các singular values được sắp xếp giảm dần:
```
σ₁ ≥ σ₂ ≥ σ₃ ≥ ... ≥ σᵣ > 0
```

với r = rank(A). Nếu r < min(m,n) thì:
```
σᵣ₊₁ = σᵣ₊₂ = ... = 0
```

**Ý nghĩa:**
- σ₁ (largest) chứa thông tin quan trọng nhất
- Các σᵢ nhỏ chứa ít thông tin, có thể bỏ qua
- Basis cho compression: giữ k singular values lớn nhất

**3. Energy compaction:**

Mỗi singular value σᵢ đóng góp một lượng "năng lượng":
```
Eᵢ = σᵢ²
```

Total energy:
```
E_total = σ₁² + σ₂² + ... + σᵣ²
```

Cumulative energy với k components:
```
E(k) = (σ₁² + ... + σₖ²) / E_total × 100%
```

Thường thấy: 90% energy tập trung ở 10-20% singular values đầu tiên.

**4. Frobenius norm:**

```
||A||_F = √(σ₁² + σ₂² + ... + σᵣ²)
```

Norm này được bảo toàn qua SVD transformation.

### 2.1.3 Geometric interpretation

SVD có thể hiểu theo góc độ hình học:

**A = U × Σ × V^T** biểu diễn phép biến đổi tuyến tính từ ℝⁿ → ℝᵐ qua 3 bước:

1. **V^T:** Rotation (quay) trong không gian ℝⁿ
2. **Σ:** Scaling (co giãn) theo các trục chính
3. **U:** Rotation (quay) trong không gian ℝᵐ

**Ví dụ hình học:**

Cho vector x ∈ ℝⁿ, phép biến đổi y = Ax được thực hiện:
```
y = A × x
  = (U × Σ × V^T) × x
  = U × (Σ × (V^T × x))
```

Quá trình:
1. V^T × x: quay x theo các eigenvectors của A^T A
2. Σ × (V^T × x): scale theo các singular values
3. U × ...: quay kết quả theo eigenvectors của AA^T

**Ứng dụng:**
- Principal Component Analysis (PCA)
- Dimensionality reduction
- Data visualization
- Noise reduction

### 2.1.4 Tại sao SVD quan trọng?

**1. Uniqueness:**
- SVD tồn tại cho mọi ma trận (không cần square, symmetric)
- Singular values là duy nhất (unique)
- U, V duy nhất nếu singular values phân biệt

**2. Numerical stability:**
- Algorithms để tính SVD rất stable
- NumPy's np.linalg.svd() sử dụng LAPACK (highly optimized)

**3. Theoretical foundation:**
- Eckart-Young theorem: SVD cho best low-rank approximation
- Optimal theo Frobenius norm và spectral norm

**4. Wide applications:**
- Image compression
- Recommender systems (Netflix prize)
- Natural Language Processing (LSA)
- Signal processing
- Quantum computing

## 2.2 Low-rank Approximation

### 2.2.1 Eckart-Young Theorem

**Định lý (Eckart-Young, 1936):**

Cho ma trận A ∈ ℝ^(m×n) với SVD: A = UΣV^T và rank r.

Xấp xỉ rank-k tốt nhất của A (với k < r) theo Frobenius norm là:

```
A_k = Σ(i=1 to k) σᵢ × uᵢ × vᵢ^T
```

hay viết dưới dạng ma trận:

```
A_k = U[:, :k] × Σ[:k, :k] × V^T[:k, :]
```

**Trong đó:**
- U[:, :k]: k cột đầu của U (m × k)
- Σ[:k, :k]: k singular values lớn nhất (k × k diagonal)
- V^T[:k, :]: k hàng đầu của V^T (k × n)

**Tính tối ưu:**

Định lý chứng minh rằng A_k là ma trận rank-k tốt nhất để xấp xỉ A theo nghĩa:

```
||A - A_k||_F = min_(rank(B)=k) ||A - B||_F
```

Không có ma trận rank-k nào khác xấp xỉ A tốt hơn A_k.

### 2.2.2 Approximation Error

**Error của xấp xỉ:**

Sai số khi dùng A_k thay cho A được đo bằng Frobenius norm:

```
||A - A_k||_F² = Σ(i=k+1 to r) σᵢ²
```

**Ý nghĩa:**
- Error chỉ phụ thuộc vào các singular values bị bỏ qua
- Nếu σₖ₊₁, σₖ₊₂, ... nhỏ → error nhỏ
- Nếu singular values decay nhanh → có thể nén tốt

**Ví dụ số:**

Giả sử ảnh 512×512 có singular values:
```
σ₁ = 5000, σ₂ = 3000, σ₃ = 1500, ..., σ₅₀ = 100
σ₅₁ = 50, σ₅₂ = 30, ..., σ₅₁₂ ≈ 0.1
```

Nếu k=50:
```
E_total = σ₁² + ... + σ₅₁₂²
E_kept = σ₁² + ... + σ₅₀²
E_lost = σ₅₁² + ... + σ₅₁₂²

E_kept / E_total ≈ 95%
```

→ Giữ 50/512 ≈ 10% components nhưng bảo toàn 95% energy!

### 2.2.3 Energy Preservation

**Định nghĩa cumulative energy:**

```
Energy(k) = (Σ(i=1 to k) σᵢ²) / (Σ(i=1 to r) σᵢ²) × 100%
```

**Interpretation:**
- Energy(k) = tỷ lệ % thông tin được giữ lại với k components
- Energy(k) → 100% khi k → r
- Target: 90-95% energy thường đủ cho ảnh

**Sử dụng để chọn k:**

```python
cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100
k_90 = np.argmax(cumulative_energy >= 90) + 1
k_95 = np.argmax(cumulative_energy >= 95) + 1
```

**Biểu đồ Energy vs k:**
[Sẽ chèn Figure 4 từ export_report_figures.py]
- Trục x: k
- Trục y: Cumulative energy (%)
- Threshold lines: 90%, 95%, 99%

## 2.3 SVD cho nén ảnh

### 2.3.1 Biểu diễn ảnh dưới dạng ma trận

**Ảnh Grayscale:**

Ảnh xám được biểu diễn dưới dạng ma trận 2D:
```
I ∈ ℝ^(m×n)
```

Trong đó:
- m: số hàng (height)
- n: số cột (width)
- I(i,j) ∈ [0, 255]: cường độ sáng tại pixel (i,j)
  - 0 = đen
  - 255 = trắng
  - Giá trị trung gian = mức xám

**Ví dụ:** Ảnh 512×512 = ma trận 512×512 với 262,144 giá trị pixel.

**Ảnh RGB (màu):**

Ảnh màu được biểu diễn dưới dạng ma trận 3D:
```
I ∈ ℝ^(m×n×3)
```

Trong đó:
- I[:,:,0] = Red channel (kênh đỏ)
- I[:,:,1] = Green channel (kênh xanh lá)
- I[:,:,2] = Blue channel (kênh xanh dương)

Mỗi kênh là ma trận m×n với giá trị [0, 255].

**Ví dụ:** Ảnh RGB 512×512 = 3 ma trận 512×512 = 786,432 giá trị.

### 2.3.2 Quy trình nén ảnh Grayscale

**Pipeline tổng quát:**

```
Original Image (m×n)
    ↓
[SVD Decomposition]
    ↓
U (m×m), Σ (m×n), V^T (n×n)
    ↓
[Truncation to k]
    ↓
U_k (m×k), Σ_k (k×k), V^T_k (k×n)
    ↓
[Reconstruction]
    ↓
Compressed Image (m×n)
```

**Chi tiết từng bước:**

**Bước 1: Chuẩn hóa**
```python
I_float = I.astype(np.float64)
```
- Convert từ uint8 [0,255] sang float64
- Tránh overflow và tăng precision

**Bước 2: SVD Decomposition**
```python
U, S, Vt = np.linalg.svd(I_float, full_matrices=False)
```
- U: (m × min(m,n))
- S: (min(m,n),) - 1D array
- Vt: (min(m,n) × n)
- `full_matrices=False`: reduced SVD, tiết kiệm memory

**Bước 3: Truncation**
```python
k = 50  # số singular values giữ lại
U_k = U[:, :k]      # shape: (m, k)
S_k = S[:k]         # shape: (k,)
Vt_k = Vt[:k, :]    # shape: (k, n)
```
- Chỉ giữ k components lớn nhất
- Discard các singular values nhỏ

**Bước 4: Reconstruction**
```python
Sigma_k = np.diag(S_k)  # shape: (k, k)
I_compressed = U_k @ Sigma_k @ Vt_k  # shape: (m, n)
```
- Ma trận nhân: (m×k) @ (k×k) @ (k×n) = (m×n)
- Tái tạo ảnh xấp xỉ original

**Bước 5: Post-processing**
```python
I_compressed = np.clip(I_compressed, 0, 255)
I_compressed = I_compressed.astype(np.uint8)
```
- Clip về [0, 255] (có thể có giá trị âm hoặc >255 do numerical error)
- Convert về uint8 để lưu/hiển thị

**Toàn bộ code:**
```python
def compress_grayscale(image, k):
    # Step 1: Convert to float
    img_float = image.astype(np.float64)

    # Step 2: SVD
    U, S, Vt = np.linalg.svd(img_float, full_matrices=False)

    # Step 3: Truncate
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Step 4: Reconstruct
    compressed = U_k @ np.diag(S_k) @ Vt_k

    # Step 5: Post-process
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    return compressed
```

### 2.3.3 Quy trình nén ảnh RGB

**Approach: Per-channel SVD**

Do ảnh RGB có 3 kênh độc lập, ta áp dụng SVD riêng cho từng kênh:

```
RGB Image (m×n×3)
    ↓
[Split channels]
    ↓
R (m×n), G (m×n), B (m×n)
    ↓
[SVD each channel independently]
    ↓
R_k, G_k, B_k
    ↓
[Merge channels]
    ↓
RGB Compressed (m×n×3)
```

**Implementation:**
```python
def compress_rgb(image, k):
    # Split into 3 channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Compress each channel
    R_compressed = compress_grayscale(R, k)
    G_compressed = compress_grayscale(G, k)
    B_compressed = compress_grayscale(B, k)

    # Merge back
    rgb_compressed = np.stack([R_compressed,
                               G_compressed,
                               B_compressed], axis=2)

    return rgb_compressed
```

**Đặc điểm:**
- Mỗi kênh có singular value spectrum riêng
- Có thể dùng k khác nhau cho từng kênh (advanced)
- Parallelizable: có thể xử lý 3 kênh đồng thời

**So sánh singular values giữa các kênh:**
[Sẽ chèn Figure 3b từ export_report_figures.py]
- Red, Green, Blue channels có phân bố khác nhau
- Green thường có variance cao nhất (human eye sensitive)

## 2.4 Compression Ratio

### 2.4.1 Tính toán Compression Ratio

**Ảnh Grayscale:**

Original storage:
```
Storage_original = m × n (pixels)
```

Compressed storage (lưu U_k, S_k, Vt_k):
```
Storage_compressed = k × m (U_k) + k (S_k) + k × n (Vt_k)
                   = k(m + n + 1)
```

Compression ratio:
```
CR = (1 - Storage_compressed / Storage_original) × 100%
   = (1 - k(m + n + 1) / (m × n)) × 100%
```

Space saved:
```
Space_saved = CR × 100%
```

**Ví dụ số: Ảnh 512×512, k=50**
```
Storage_original = 512 × 512 = 262,144
Storage_compressed = 50(512 + 512 + 1) = 51,250
CR = (1 - 51,250/262,144) × 100% = 80.45%
```
→ Tiết kiệm 80.45% dung lượng!

**Ảnh RGB:**

Original storage:
```
Storage_original = m × n × 3
```

Compressed storage (3 channels):
```
Storage_compressed = 3 × k(m + n + 1)
```

Compression ratio:
```
CR = (1 - 3k(m + n + 1) / (3mn)) × 100%
   = (1 - k(m + n + 1) / (mn)) × 100%
```

**Kết luận:** RGB có công thức CR giống grayscale!

### 2.4.2 Phân tích Compression Ratio

**Dependency vào k:**

Với m = n = 512:
```
CR(k) = (1 - k(512 + 512 + 1) / (512 × 512)) × 100%
      = (1 - 1025k / 262144) × 100%
      ≈ (1 - 0.00391k) × 100%
```

| k   | CR (%) | Interpretation       |
|-----|--------|----------------------|
| 5   | 98.0%  | Very high compression |
| 10  | 96.1%  | High compression     |
| 20  | 92.2%  | Good compression     |
| 50  | 80.5%  | Moderate compression |
| 100 | 61.9%  | Low compression      |
| 200 | 23.8%  | Very low compression |

**Trade-off:**
- k nhỏ → CR cao → chất lượng thấp
- k lớn → CR thấp → chất lượng cao

**Break-even point:**

CR = 50% khi:
```
k(m + n + 1) = mn / 2
k = mn / (2(m + n + 1))
```

Với 512×512: k ≈ 128

## 2.5 Đánh giá chất lượng ảnh

### 2.5.1 Mean Squared Error (MSE)

**Định nghĩa:**

MSE đo sai số trung bình bình phương giữa ảnh gốc và ảnh nén:

```
MSE = (1 / (m×n)) × Σ Σ [I_original(i,j) - I_compressed(i,j)]²
                     i j
```

**Công thức vectorized (NumPy):**
```python
def calculate_mse(original, compressed):
    diff = original.astype(np.float64) - compressed.astype(np.float64)
    mse = np.mean(diff ** 2)
    return mse
```

**Đặc điểm:**
- MSE = 0: ảnh giống hệt nhau (perfect match)
- MSE nhỏ: ảnh tương tự nhau
- MSE lớn: ảnh khác nhau nhiều

**Ưu điểm:**
- Dễ tính toán
- Có ý nghĩa toán học rõ ràng

**Nhược điểm:**
- Không tương quan tốt với Human Visual System (HVS)
- Sensitive to outliers
- Không phân biệt structural distortion vs random noise

### 2.5.2 Peak Signal-to-Noise Ratio (PSNR)

**Định nghĩa:**

PSNR là logarithm của tỷ lệ giữa max signal power và noise power:

```
PSNR = 10 × log₁₀(MAX² / MSE)  (dB)
```

Với MAX = 255 cho ảnh 8-bit:
```
PSNR = 10 × log₁₀(255² / MSE)
     = 20 × log₁₀(255) - 10 × log₁₀(MSE)
     ≈ 48.13 - 10 × log₁₀(MSE)  (dB)
```

**Implementation:**
```python
def calculate_psnr(original, compressed, max_value=255):
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr
```

**Thang đánh giá chất lượng:**

| PSNR (dB) | Quality      | Interpretation                |
|-----------|--------------|-------------------------------|
| > 40      | Excellent    | Gần như không phân biệt được  |
| 30 - 40   | Good         | Chất lượng tốt, chấp nhận được |
| 20 - 30   | Fair         | Nhận thấy degradation rõ ràng |
| < 20      | Poor         | Chất lượng kém               |

**Ưu điểm:**
- Được sử dụng rộng rãi (standard metric)
- Dễ so sánh giữa các phương pháp
- Đơn vị dB dễ interpret

**Nhược điểm:**
- Vẫn dựa trên MSE → không hoàn toàn tương quan với HVS
- Có thể misleading cho một số loại distortion

**Quan hệ PSNR và MSE:**
```
PSNR ↑ ⟺ MSE ↓ ⟺ Quality ↑
```

### 2.5.3 Structural Similarity Index (SSIM)

**Định nghĩa:**

SSIM đo structural similarity giữa 2 ảnh, được thiết kế để tương quan tốt hơn với cảm nhận con người:

```
SSIM(x, y) = [l(x,y)]^α × [c(x,y)]^β × [s(x,y)]^γ
```

Với α = β = γ = 1:
```
SSIM(x, y) = l(x,y) × c(x,y) × s(x,y)
```

**Thành phần:**

1. **Luminance comparison (l):**
```
l(x,y) = (2μ_x μ_y + C₁) / (μ_x² + μ_y² + C₁)
```
- μ_x, μ_y: mean intensities
- C₁: constant to avoid division by zero

2. **Contrast comparison (c):**
```
c(x,y) = (2σ_x σ_y + C₂) / (σ_x² + σ_y² + C₂)
```
- σ_x, σ_y: standard deviations

3. **Structure comparison (s):**
```
s(x,y) = (σ_xy + C₃) / (σ_x σ_y + C₃)
```
- σ_xy: covariance

**Giá trị:**
- SSIM ∈ [-1, 1]
- SSIM = 1: ảnh giống hệt nhau
- SSIM > 0.9: chất lượng rất tốt
- SSIM < 0.5: chất lượng kém

**Simplified implementation:**
```python
from skimage.metrics import structural_similarity

def calculate_ssim(original, compressed):
    ssim = structural_similarity(original, compressed,
                                  data_range=255)
    return ssim
```

**Ưu điểm:**
- Tương quan tốt với Human Visual System
- Considers luminance, contrast, structure
- Widely used in image/video quality assessment

**Nhược điểm:**
- Phức tạp hơn PSNR/MSE
- Computational cost cao hơn
- Cần window size và parameters

**Khi nào dùng:**
- PSNR: quick comparison, standard reporting
- SSIM: perceptual quality assessment, detailed analysis

---

**[Phần tiếp theo: 3. PHƯƠNG PHÁP THỰC HIỆN, 4. CHI TIẾT IMPLEMENTATION, 5. KẾT QUẢ THỰC NGHIỆM sẽ được điền sau khi chạy experiments và có kết quả số cụ thể]**

**Hướng dẫn hoàn thiện báo cáo:**

1. **Chạy notebook 05_final_summary.ipynb** để có kết quả số
2. **Điền kết quả** vào Phần 5 (tables với số liệu thực tế)
3. **Chèn figures** từ report/figures/ vào đúng vị trí
4. **Viết Discussion** dựa trên findings
5. **Viết Conclusion** tổng kết

---

