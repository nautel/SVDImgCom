# Presentation Outline
# Nén Ảnh Bằng Phương Pháp SVD

**Thời lượng:** 15-20 phút
**Slides:** 15-20 slides

---

## Slide 1: Title Slide (30 seconds)


## Slide 2: Agenda (30 seconds)

**Content:**
1. Bối cảnh và động lực
2. Lý thuyết SVD
3. Phương pháp thực hiện
4. Kết quả thực nghiệm
5. Kết luận và hướng phát triển

**Visual:**
- Bullet list với icons
- Highlight current section marker

**Speaker notes:**
"Bài trình bày gồm 5 phần chính như sau..."

---

## Slide 3: Bối cảnh - Tại sao cần nén ảnh? (1 phút)

**Content:**
**Vấn đề:**
- 3.2 tỷ ảnh được share mỗi ngày
- Dung lượng lưu trữ hạn chế
- Băng thông internet giới hạn

**Ứng dụng:**
- Y tế: X-ray, MRI scans
- Viễn thám: Ảnh vệ tinh
- Web/Mobile: Website, apps
- Backup và cloud storage

**Visual:**
- Statistics với số liệu lớn
- Icons cho từng ứng dụng
- Biểu đồ tăng trưởng dữ liệu ảnh

**Speaker notes:**
"Với sự bùng nổ của dữ liệu ảnh, nén ảnh trở nên vô cùng quan trọng..."

---

## Slide 4: Vấn đề nghiên cứu (1 phút)

**Content:**
**Câu hỏi chính:**
> Làm thế nào để giảm dung lượng ảnh mà vẫn giữ được chất lượng?

**Trade-off:**
```
High Quality ←→ High Compression
```

**Mục tiêu:**
- Áp dụng SVD để nén ảnh
- Đánh giá hiệu quả (PSNR, compression ratio)
- Tìm k tối ưu

**Visual:**
- Balance scale: Quality vs Compression
- Highlight trade-off
- Goal icons

**Speaker notes:**
"Đây là bài toán optimization với 2 mục tiêu đối lập..."

---

## Slide 5: SVD là gì? (2 phút)

**Content:**
**Singular Value Decomposition:**

```
A = U × Σ × V^T
```

**Thành phần:**
- **U** (m×m): Left singular vectors
- **Σ** (m×n): Singular values (diagonal)
- **V^T** (n×n): Right singular vectors

**Đặc điểm:**
- U, V là ma trận trực giao
- Singular values sắp xếp giảm dần: σ₁ ≥ σ₂ ≥ ...
- Tối ưu cho low-rank approximation

**Visual:**
- Animated diagram: A = U × Σ × V^T
- Matrix visualization (use fig2 from export script)
- Color-code từng thành phần

**Speaker notes:**
"SVD phân tích ma trận A thành tích của 3 ma trận. U và V là trực giao, Σ chứa singular values..."

---

## Slide 6: Low-Rank Approximation (1.5 phút)

**Content:**
**Eckart-Young Theorem:**

Xấp xỉ tốt nhất rank-k:
```
A_k = U[:,:k] × Σ[:k,:k] × V^T[:k,:]
```

**Dung lượng:**
- Original: m × n
- Compressed: k(m + n + 1)
- Compression ratio: (1 - k(m+n+1)/(mn)) × 100%

**Ví dụ:** Ảnh 512×512, k=50
- Original: 262,144 values
- Compressed: 51,250 values
- **Saved: 80.5%!**

**Visual:**
- Formula visualization
- Storage comparison bar chart
- Example calculation highlighted

**Speaker notes:**
"Định lý Eckart-Young chứng minh rằng SVD cho xấp xỉ rank-k tối ưu nhất..."

---

## Slide 7: SVD cho nén ảnh (1.5 phút)

**Content:**
**Pipeline:**
```
Original (m×n)
    ↓ SVD
U, Σ, V^T
    ↓ Keep top k
U_k, Σ_k, V^T_k
    ↓ Reconstruct
Compressed (m×n)
```

**Grayscale:** 1 channel → 1 SVD
**RGB:** 3 channels → 3 SVDs (per-channel)

**Visual:**
- Flowchart với từng bước
- Before/after images (k=10 vs k=50)
- Use fig1 (original images)

**Speaker notes:**
"Quy trình nén gồm 3 bước: decomposition, truncation, reconstruction..."

---

## Slide 8: Đánh giá chất lượng (1 phút)

**Content:**
**Metrics:**

1. **PSNR** (Peak Signal-to-Noise Ratio)
   ```
   PSNR = 10 × log₁₀(255²/MSE) dB
   ```
   - > 40 dB: Excellent
   - 30-40 dB: Good ✅
   - < 30 dB: Fair/Poor

2. **MSE** (Mean Squared Error)
   - Lower is better

3. **SSIM** (Structural Similarity)
   - Range [0, 1], closer to 1 is better

**Visual:**
- Metric formulas
- Quality thresholds table
- Color-coded ranges

**Speaker notes:**
"Chúng ta đánh giá chất lượng qua 3 metrics chính: PSNR, MSE, và SSIM..."

---

## Slide 9: Implementation (1 phút)

**Content:**
**Tools:**
- Python 3.8+
- NumPy, SciPy, Matplotlib, OpenCV
- Jupyter Notebooks

**Modules:**
1. `image_utils.py` - Load/save ảnh
2. `svd_compression.py` - Core algorithms
3. `quality_metrics.py` - PSNR, MSE, SSIM
4. `visualization.py` - Charts và plots

**Test images:**
- Grayscale: 5.3.01.tiff (512×512)
- RGB: 4.1.01.tiff (512×512×3)

**Visual:**
- Code structure diagram
- Library logos
- Test images preview

**Speaker notes:**
"Hệ thống được triển khai bằng Python với 4 modules chính..."

---

## Slide 10: Kết quả - Grayscale (2 phút)

**Content:**
**Test image:** 5.3.01.tiff (512×512)

| k   | PSNR (dB) | SSIM   | Saved  |
|-----|-----------|--------|--------|
| 10  | ~28       | ~0.85  | 96.1%  |
| 20  | ~34       | ~0.94  | 92.2%  |
| 50  | ~41       | ~0.98  | 80.5%  |
| 100 | ~47       | ~0.99  | 61.9%  |

**Key findings:**
- k=20: Good quality (PSNR > 30), high compression
- k=50: Excellent quality (PSNR > 40)

**Visual:**
- Use fig5_grayscale_compression.png (comparison grid)
- Highlight optimal k=20-50
- PSNR vs k chart (fig7)

**Speaker notes:**
"Với ảnh grayscale 512×512, kết quả cho thấy k=20-50 là điểm cân bằng tốt..."

---

## Slide 11: Kết quả - RGB (1.5 phút)

**Content:**
**Test image:** 4.1.01.tiff (512×512×3)

| k   | PSNR (dB) | Saved  | Quality   |
|-----|-----------|--------|-----------|
| 10  | ~26       | 96.1%  | Fair      |
| 20  | ~32       | 92.2%  | Good ✅   |
| 50  | ~39       | 80.5%  | Excellent |

**Per-channel SVD:**
- R, G, B channels compressed independently
- Same compression ratio as grayscale
- Color information preserved

**Visual:**
- Use fig6_rgb_compression.png
- RGB channels visualization (fig from notebook 03)
- Side-by-side comparison

**Speaker notes:**
"Ảnh RGB được xử lý bằng per-channel SVD, kết quả tương tự grayscale..."

---

## Slide 12: Singular Value Analysis (1.5 phút)

**Content:**
**Singular Value Spectrum:**
- Rapid decay → good for compression
- Top 50 values capture > 90% energy

**Cumulative Energy:**
- 90% energy: k ≈ 30-40
- 95% energy: k ≈ 50-60
- 99% energy: k ≈ 100+

**Visual:**
- Use fig3_singular_value_spectrum.png
- Use fig4_cumulative_energy.png
- Annotate 90%, 95% thresholds

**Speaker notes:**
"Phân tích singular values cho thấy sự suy giảm nhanh, hỗ trợ việc nén hiệu quả..."

---

## Slide 13: Trade-off Analysis (2 phút)

**Content:**
**Quality vs Compression:**

```
High k → High quality, Low compression
Low k → Low quality, High compression
```

**Sweet spot: k = 20-50**
- PSNR: 30-40 dB (good quality)
- Compression: 80-92% saved
- Use case: General storage, web images

**Visual:**
- Use fig8_tradeoff.png
- Annotate sweet spot region
- Error maps (fig9) for k=10, 20, 50

**Speaker notes:**
"Biểu đồ trade-off cho thấy rõ mối quan hệ giữa chất lượng và nén. Sweet spot nằm ở k=20-50..."

---

## Slide 14: So sánh Grayscale vs RGB (1 phút)

**Content:**
**Findings:**

| Aspect | Grayscale | RGB |
|--------|-----------|-----|
| Compression ratio | Same for same k | Same for same k |
| Processing time | Faster (1 channel) | Slower (3 channels) |
| Information | Luminance only | Full color |
| Use case | Documents, medical | Natural images, photos |

**Recommendation:**
- Grayscale: Simpler, faster, sufficient for B&W content
- RGB: Necessary for color images, worth the extra cost

**Visual:**
- Comparison table
- Example images side-by-side
- Use fig from notebook 04 (comparison)

**Speaker notes:**
"So sánh cho thấy compression ratio giống nhau, nhưng RGB cần xử lý 3× data..."

---

## Slide 15: Ưu điểm và Hạn chế (1 phút)

**Content:**
**Ưu điểm:**
✅ Toán học vững chắc (Eckart-Young)
✅ Dễ hiểu và implement
✅ Linh hoạt (chọn k tùy ý)
✅ Tốt cho mục đích học tập

**Hạn chế:**
❌ Phức tạp tính toán O(mn²)
❌ Không hiệu quả bằng JPEG
❌ Cần lưu 3 ma trận U, Σ, V^T
❌ Không tận dụng local correlations

**Visual:**
- Two-column layout (pros vs cons)
- Icons for each point
- Complexity comparison chart

**Speaker notes:**
"SVD có nhiều ưu điểm về mặt lý thuyết, nhưng không cạnh tranh được với JPEG trong thực tế..."

---

## Slide 16: So sánh với JPEG (1 phút)

**Content:**
| Tiêu chí | SVD | JPEG |
|----------|-----|------|
| Compression ratio | 70-90% | 90-98% |
| Speed | Slow (O(mn²)) | Fast |
| Quality | Good, no blocking | Excellent, blocking at low quality |
| Use case | Educational, research | Industry standard |

**Kết luận:**
- JPEG tốt hơn cho production
- SVD tốt cho học tập và nghiên cứu

**Visual:**
- Comparison table
- Sample compressed images (SVD vs JPEG)
- Blocking artifacts demonstration

**Speaker notes:**
"JPEG vượt trội hơn SVD về mặt thực tế, nhưng SVD có giá trị học tập lớn..."

---

## Slide 17: Kết luận (1 phút)

**Content:**
**Tổng kết:**
- ✅ Triển khai thành công SVD compression
- ✅ Đạt 80-92% compression với PSNR ≥ 30dB
- ✅ Tìm được k tối ưu: 20-50 cho 512×512
- ✅ Phân tích toàn diện với 30+ visualizations

**Bài học:**
- Hiểu sâu về SVD và linear algebra
- Kỹ năng Python và data visualization
- Trade-off trong image compression
- Scientific research methodology

**Visual:**
- Checkmarks với achievements
- Summary statistics
- Project timeline

**Speaker notes:**
"Dự án đã đạt được các mục tiêu đề ra và mang lại nhiều bài học quý giá..."

---

## Slide 18: Hướng phát triển (1 phút)

**Content:**
**Future Work:**

**Short-term:**
- Test trên dataset lớn hơn (20+ images)
- Block-based SVD (8×8 blocks như JPEG)
- Benchmark với JPEG compression

**Long-term:**
- Adaptive k selection algorithm
- YCbCr color space
- GPU acceleration
- Real-time compression
- GUI application

**Visual:**
- Roadmap timeline
- Icons cho từng feature
- Priority markers

**Speaker notes:**
"Có nhiều hướng phát triển thú vị từ đề tài này..."

---

## Slide 19: Demo (Optional) (2 phút)

**Content:**
**Live Demo:**
1. Load ảnh
2. Compress với k khác nhau
3. Show PSNR và compression ratio
4. Visual comparison

**Backup:** Screenshots nếu demo fail

**Visual:**
- Jupyter notebook running
- hoặc screenshots của kết quả

**Speaker notes:**
"Bây giờ em xin demo nhanh quá trình nén ảnh..."

---

## Slide 20: Q&A (Dự phòng)

**Content:**
**Câu hỏi thường gặp:**

Q: Tại sao không dùng JPEG?
A: SVD có giá trị học tập, dễ hiểu về mặt toán học

Q: k tối ưu là bao nhiêu?
A: Phụ thuộc use case, thường k=20-50 cho 512×512

Q: SVD có thể real-time không?
A: Không, do complexity O(mn²). Cần GPU hoặc approximate methods.

**Visual:**
- FAQ list
- Contact info
- Thank you message

**Speaker notes:**
"Em xin chân thành cảm ơn thầy/cô và các bạn đã lắng nghe!"

---

## Slide 21: Thank You + Contact (30 seconds)

**Content:**
**Thank You!**

**Tài liệu:**
- GitHub: [Link nếu có]
- Email: [Email sinh viên]
- Report: [Path to report]

**Tài liệu tham khảo:**
- Gilbert Strang - "Introduction to Linear Algebra"
- Steve Brunton - SVD YouTube Series
- USC-SIPI Image Database

**Visual:**
- Large "Thank You" text
- QR code to GitHub (if available)
- Contact information

**Speaker notes:**
"Em xin chân thành cảm ơn! Nếu có câu hỏi, xin mời thầy/cô và các bạn!"

---

## Phụ lục: Tips cho Presentation

### Chuẩn bị:
1. **Rehearse 2-3 lần** - Tổng thời gian 15-18 phút
2. **Backup slides** dưới dạng PDF
3. **Test demo trước** - Có screenshots backup
4. **In handouts** nếu cần

### Delivery:
- **Speak clearly** và không quá nhanh
- **Eye contact** với audience
- **Point to visuals** khi giải thích
- **Pause** sau mỗi key point

### Anticipate Questions:
- Tại sao chọn SVD?
- So sánh với JPEG chi tiết hơn?
- Complexity analysis
- Practical applications

### Time Management:
- **1-5 phút:** Introduction + Background
- **6-10 phút:** Theory + Method
- **11-16 phút:** Results + Analysis
- **17-20 phút:** Conclusion + Q&A

### Visual Guidelines:
- **Font size ≥ 24pt** cho text
- **Max 6 bullets** per slide
- **High contrast** colors
- **Consistent** theme

---

**Good luck with your presentation! 🎤✨**
