# 🚀 Quick Start Guide - SVD Image Compression

**Get started in 5 minutes!**

---

## ⚡ Fast Track (3 Steps)

### Step 1: Setup Environment

```bash
# Navigate to project
cd C:\Users\nuate\project

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_svd_theory_demo.ipynb
# 2. notebooks/02_grayscale_compression.ipynb
# 3. notebooks/03_color_compression.ipynb
# 4. notebooks/04_comparative_analysis.ipynb
# 5. notebooks/05_final_summary.ipynb
```

### Step 3: View Results

```
results/
├── compressed/        # Compressed images
├── visualizations/    # Charts and plots
└── metrics/          # CSV/JSON data
```

**Done! 🎉**

---

## 📝 Quick Examples

### Example 1: Compress a Grayscale Image

```python
import sys
sys.path.append('src')

from image_utils import load_image, save_image
from svd_compression import compress_grayscale
from quality_metrics import calculate_psnr

# Load image
img = load_image('images/grayscale/5.3.01.tiff', mode='GRAY')

# Compress with k=50
compressed = compress_grayscale(img, k=50)

# Check quality
psnr = calculate_psnr(img, compressed)
print(f"PSNR: {psnr:.2f} dB")

# Save result
save_image(compressed, 'results/compressed/my_compressed.png')
```

**Output:**
```
PSNR: 41.23 dB  (Excellent quality!)
```

### Example 2: Compress an RGB Image

```python
from svd_compression import compress_rgb

# Load color image
img_rgb = load_image('images/color/4.1.01.tiff', mode='RGB')

# Compress with k=30
compressed_rgb = compress_rgb(img_rgb, k=30)

# Check quality
psnr = calculate_psnr(img_rgb, compressed_rgb)
print(f"PSNR: {psnr:.2f} dB")
```

### Example 3: Test Multiple k Values

```python
from quality_metrics import calculate_all_metrics
from svd_compression import calculate_compression_ratio

k_values = [10, 20, 50, 100]

for k in k_values:
    comp = compress_grayscale(img, k)
    metrics = calculate_all_metrics(img, comp)
    stats = calculate_compression_ratio(img.shape, k, is_rgb=False)

    print(f"k={k:3d} | PSNR={metrics['psnr']:6.2f}dB | "
          f"Saved={stats['space_saved_percent']:5.1f}%")
```

**Output:**
```
k= 10 | PSNR= 28.45dB | Saved= 96.1%
k= 20 | PSNR= 34.12dB | Saved= 92.2%
k= 50 | PSNR= 41.23dB | Saved= 80.5%
k=100 | PSNR= 47.56dB | Saved= 61.9%
```

---

## 🎯 Common Tasks

### Task 1: Find Optimal k

```python
from svd_compression import calculate_cumulative_energy, get_svd_matrices
import numpy as np

# Get singular values
U, S, Vt = get_svd_matrices(img)

# Calculate energy
cumulative_energy = calculate_cumulative_energy(S)

# Find k for 90% energy
k_90 = np.argmax(cumulative_energy >= 90) + 1
print(f"k for 90% energy: {k_90}")

# Find k for 95% energy
k_95 = np.argmax(cumulative_energy >= 95) + 1
print(f"k for 95% energy: {k_95}")
```

### Task 2: Create Visualizations

```python
from visualization import (
    plot_singular_values,
    plot_cumulative_energy,
    plot_quality_vs_k
)

# Plot singular value spectrum
U, S, Vt = get_svd_matrices(img)
plot_singular_values(S,
                     title="Singular Values",
                     save_path='my_spectrum.png')

# Plot energy preservation
plot_cumulative_energy(S,
                       save_path='my_energy.png')
```

### Task 3: Export Report Figures

```bash
# Generate all 9 high-res figures for report
python export_report_figures.py
```

**Output:**
- `report/figures/fig1_original_images.png`
- `report/figures/fig2_svd_decomposition.png`
- `report/figures/fig3_singular_value_spectrum.png`
- `report/figures/fig4_cumulative_energy.png`
- `report/figures/fig5_grayscale_compression.png`
- `report/figures/fig6_rgb_compression.png`
- `report/figures/fig7_quality_metrics.png`
- `report/figures/fig8_tradeoff.png`
- `report/figures/fig9_error_maps.png`

---

## 📊 Understanding Results

### Quality Thresholds

| PSNR Range | Quality   | Use Case                  |
|------------|-----------|---------------------------|
| > 40 dB    | Excellent | Archival, medical imaging |
| 30-40 dB   | Good      | General storage           |
| 20-30 dB   | Fair      | Web thumbnails            |
| < 20 dB    | Poor      | Not recommended           |

### Compression Guidelines

For **512×512 images**:

| k    | Compression | PSNR (approx) | Recommendation       |
|------|-------------|---------------|----------------------|
| 5-10 | Very High   | 20-28 dB      | Previews only        |
| 20-30| High        | 30-36 dB      | ⭐ **Recommended**   |
| 50-70| Moderate    | 38-42 dB      | High quality storage |
| 100+ | Low         | 45+ dB        | Near-lossless        |

**Rule of thumb:**
```
k ≈ 0.05 - 0.10 × min(width, height)
```

For 512×512: k = 25-50

---

## 🔧 Troubleshooting

### Issue 1: ModuleNotFoundError

```bash
# Make sure you're in project directory
cd C:\Users\nuate\project

# Reinstall requirements
pip install -r requirements.txt
```

### Issue 2: Image not found

```python
# Use absolute path
import os
img_path = os.path.join(os.getcwd(), 'images/grayscale/5.3.01.tiff')
img = load_image(img_path)
```

### Issue 3: PSNR = inf

This is normal when MSE = 0 (perfect match), usually when k = min(m, n).

### Issue 4: Low PSNR with high k

- Check image loaded correctly
- Verify k < min(m, n)
- Try different k values

---

## 🎓 Learning Path

**Beginner (Day 1):**
1. Read README.md
2. Run notebook 01 (SVD theory)
3. Try Example 1 (grayscale compression)

**Intermediate (Day 2):**
1. Run notebooks 02-04
2. Experiment with different k values
3. Create your own visualizations

**Advanced (Day 3):**
1. Run notebook 05 (final summary)
2. Export report figures
3. Write report using outline

---

## 📚 Next Steps

1. **Explore the code:**
   ```bash
   # Read module docstrings
   python -c "import src.svd_compression; help(src.svd_compression)"
   ```

2. **Test with your images:**
   - Place images in `images/` folder
   - Modify notebook 02/03 to use your images

3. **Complete the report:**
   - Use `report/report_outline.md` as template
   - Fill in your experimental results
   - Insert figures from `report/figures/`

4. **Prepare presentation:**
   - Use visualizations from `results/visualizations/`
   - Focus on key findings (optimal k, trade-offs)

---

## 💡 Tips

- **Start with notebook 05** for a complete demo
- **k=20-50** usually gives best balance
- **PSNR ≥ 30dB** is considered good quality
- **Run export_report_figures.py** before writing report
- **Check cumulative energy** to choose optimal k

---

## 🆘 Need Help?

1. **Check documentation:**
   - README.md (comprehensive guide)
   - Notebooks (step-by-step demos)
   - Docstrings in code

2. **Review examples:**
   - test_user_images.py (complete example)
   - Module __main__ blocks (self-tests)

3. **Common fixes:**
   - Restart Jupyter kernel
   - Reinstall dependencies
   - Check file paths

---

**Happy Compressing! 🎉**

Questions? Check README.md or notebook comments.
