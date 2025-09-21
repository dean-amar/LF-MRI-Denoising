# **LF-MRI Denoising (NLM)**
Practical guide for running `solver.py`, understanding the **Non-Local Means (NLM)** algorithm, and configuring parameters via **config.yaml** and **CLI flags**.

## 1. What is Non-Local Means (NLM)?
**Non-Local Means (NLM)** is an edge-preserving denoising algorithm.  
Unlike simple blurring, which averages only neighboring pixels, NLM averages **patches** across the entire search window if they are **similar** to the current patch.

### Algorithm intuition
- Noise is random, but structures repeat.  
- Similar patches in different parts of the image reinforce each other when averaged.  
- Result: **noise reduction** with **edges and textures preserved**.

### Core hyper-parameters
- **h (filter strength):** Larger → stronger denoising, risk of oversmoothing.  
- **small_window_size**: Patch size. Larger → more context per patch, but more smoothing.  
- **big_window_size**: Search window. Larger → more candidate patches, slower runtime. 

---

## 2. Virtual Environment Setup

This repository includes a prepared **virtual environment setup**.  
```bash
# 1. Clone the repo
git clone https://github.com/dean-amar/LF-MRI-Denoising.git
cd LF-MRI-Denoising

# 2. Create a virtual environment (named .venv)
python3 -m venv .venv

# 3. Activate the environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```
---

## 3. Running `solver.py`
There are **two main modes**:
### A) Process the entire dataset
```bash
# Example
python solver.py --h=7 --small_window_size=3 --big_window_size=5 --process_all=true --save=true
```
### B) Process a single image
```bash
# Example
python solver.py --h=7 --small_window_size=3 --big_window_size=5 --single=true --save=true
```
If `--single=true`, the solver takes the one image specified in `config.yaml.`

---

## 4. Parameter
You can configure values in config.yaml and override them via CLI flags.

CLI always takes precedence.

### CLI flags

`--h`: Smoothing parameter

`--small_window_size`: Size of the processing window

`--big_window_size`: Size of the search window

`--plot`: Plot image output

`--save`: Save denoised image

`--single`: Process single image

`--process_all`: Process all dataset

---

## 5. Output
When you run `solver.py`, the program can generate several types of output depending on the parameters you choose:

### Processed images
All denoised images are saved into the folder specified in `SAVE_PATH` (from `config.yaml`).  
If `--save=false`, no images are written to disk.

### Visualization
If you set `--plot=true`, the processed image will also be displayed on screen using matplotlib.  
This is useful for quickly inspecting the effect of different parameters without saving files.

---

## 6. Tips
To get the best results from the Non-Local Means denoising:

- Begin with **moderate settings** such as `--h=9..11`, `--small_window_size=5`, and `--big_window_size=21..33`.  
  These provide a good balance between noise reduction and detail preservation.

- If the images still look noisy, **increase `h`** gradually.  
  Be careful: very high values may oversmooth anatomical structures.

- If fine details look blurred, **reduce `h`** or try smaller patch and search windows.  
  This preserves more detail but may leave residual noise.

- Remember that **large search windows (≥70)** provide stronger smoothing but at the cost of significantly slower runtime.  
  They should be used only when maximum denoising is required and computational time is not a concern.

