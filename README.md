# Flux Kontext CFD Surrogate Model

A project for building a FLUX.1-Kontext-dev based surrogate model for Computational Fluid Dynamics (CFD) using the AirfRANS dataset. This project trains LoRA adapters on preprocessed CFD contour images to enable rapid flow field predictions and parameter studies.

## Project Overview

This project aims to create a surrogate model for CFD simulations by training FLUX.1-Kontext-dev on paired CFD contour images. The model learns to predict flow field visualizations (pressure, velocity components, turbulence) given different airfoil geometries, angles of attack, and flow conditions.

### Key Features

- **CFD Surrogate Modeling**: Train LoRA adapters on AirfRANS dataset for rapid flow field predictions
- **Multi-variable Support**: Handle pressure, velocity components (u_x, u_y), and turbulence (nut) visualizations
- **Parameter Studies**: Enable quick exploration of airfoil geometry, angle of attack, and velocity variations
- **Automated Dataset Generation**: Preprocess raw CFD images into training pairs with proper captions
- **Comprehensive Analysis**: Compare generated results against ground truth with multiple metrics

## Setup

### Local Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Important**: Install additional required dependencies:
```bash
pip install protobuf sentencepiece
```

4. **Hugging Face Authentication** (Required for gated model access):
```bash
hf auth login
```
You'll need a Hugging Face token with access to the FLUX.1-Kontext-dev model. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Lambda AI Cluster Setup

If you're setting up on a Lambda AI cluster or similar GPU cloud instance:

1. **SSH into your cluster**:
```bash
ssh ubuntu@<your-cluster-ip>
```

2. **Clone the repository**:
```bash
git clone https://github.com/krishnanj/flux_kontext_lyrics.git
cd flux_kontext_lyrics
```

3. **Set up Python environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install protobuf sentencepiece
```

5. **Authenticate with Hugging Face**:
```bash
hf auth login
```

6. **Configure Git** (for pushing results):
```bash
git config --global user.email "your-email@example.com"
git config --global user.name "your-username"
```

## Dataset Preprocessing

The project processes raw CFD contour images into training pairs suitable for FLUX.1-Kontext-dev training. The preprocessing focuses on three types of meaningful transformations:

### 1. Airfoil Geometry Changes
- **Purpose**: Learn how different airfoil shapes affect flow patterns
- **Method**: Pair images with different airfoil geometries under similar flow conditions
- **Example**: NACA0012 → NACA2412 at same angle of attack and velocity
- **Caption**: "fludyn edit: change airfoil from NACA0012 to NACA2412; keep variable=pressure, AoA≈5.0, V≈50.0"

### 2. Angle of Attack (AoA) Changes
- **Purpose**: Understand how flow separation and pressure distribution change with AoA
- **Method**: Pair images with different AoA values for the same airfoil and variable
- **Example**: Same airfoil at 0° → 10° angle of attack
- **Caption**: "fludyn edit: set AoA from 0.00 to 10.00 deg; keep airfoil=NACA0012, variable=pressure"

### 3. Velocity Changes
- **Purpose**: Learn Reynolds number effects on flow patterns
- **Method**: Pair images with different velocities for same airfoil and AoA
- **Example**: Same airfoil at 30 m/s → 60 m/s
- **Caption**: "fludyn edit: set V from 30.000 to 60.000; keep airfoil=NACA0012, variable=pressure, AoA≈5.0"

### Excluded Transformations
- **Variable Switching**: We explicitly exclude pairs that switch between variables (e.g., pressure → velocity) for the same conditions, as this is not useful for surrogate modeling since all variables are computed together in CFD solvers.

## Usage

### 1. Dataset Generation

Generate training datasets from raw CFD images:

```bash
python build_kontext_ds.py --raw-root plots --out-root kontext_run_small_new --max-pairs 20 --seed 42
```

### 2. Training LoRA Adapters

Train on Lambda AI cluster or local GPU:

```bash
# On cluster
cd ai-toolkit
python run.py /path/to/kontext_lora_config.yaml

# With background execution
nohup ./run_kontext_training.sh &
```

### 3. Analysis and Evaluation

Compare generated samples against ground truth:

```bash
python analyze_samples.py \
  --samples output_fludyn/samples \
  --gt kontext_run_small_new/test_results \
  --out output_fludyn/analysis \
  --save-diffs
```

## Expected Results and Analysis

### Training Outcomes

The trained LoRA adapters should learn to:
- **Predict flow patterns** for new airfoil geometries within the training parameter space
- **Extrapolate flow behavior** for angles of attack and velocities not seen during training
- **Maintain physical consistency** in pressure distributions, velocity fields, and turbulence patterns
- **Generate realistic CFD visualizations** that match the style and quality of the training data

### Analysis Methodology

We use multiple metrics to evaluate the surrogate model performance:

#### 1. Pixel-level Metrics
- **MSE (Mean Squared Error)**: Direct pixel-wise comparison
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality assessment
- **SSIM (Structural Similarity Index)**: Structural similarity preservation
- **MAE (Mean Absolute Error)**: Robust pixel difference measure

#### 2. Perceptual Metrics
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Human-perceived similarity using AlexNet features

#### 3. Visual Analysis
- **Difference Heatmaps**: Highlight regions of maximum deviation
- **Side-by-side Comparisons**: Generated vs ground truth visualizations
- **Error Distribution Analysis**: Identify systematic biases or failure modes

### Success Criteria

A successful surrogate model should achieve:
- **SSIM > 0.8**: High structural similarity to ground truth
- **LPIPS < 0.2**: Low perceptual difference
- **PSNR > 25 dB**: Good image quality
- **Physical Consistency**: Generated flow patterns should follow fluid dynamics principles

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended: 24GB+ VRAM for training)
- Hugging Face account with access to FLUX.1-Kontext-dev model
- Additional dependencies: `torchmetrics`, `lpips`, `matplotlib` (for analysis)

## Model

This script uses the `black-forest-labs/FLUX.1-Kontext-dev` model from Hugging Face, which is a **gated repository**. You must:

1. Have a Hugging Face account
2. Request access to the FLUX.1-Kontext-dev model
3. Be authenticated with `hf auth login`

The model will be automatically downloaded on first use (approximately 20GB).

## Troubleshooting

### Common Issues

1. **"Cannot access gated repo" error**:
   - Make sure you're authenticated with `hf auth login`
   - Verify you have access to the FLUX.1-Kontext-dev model
   - Check your Hugging Face token permissions

2. **"protobuf library not found" error**:
   - Install protobuf: `pip install protobuf sentencepiece`

3. **"Cannot instantiate tokenizer" error**:
   - Install sentencepiece: `pip install sentencepiece`

4. **Memory issues**:
   - The model requires significant GPU memory (recommended: 24GB+ VRAM)
   - Consider using CPU mode for smaller images if GPU memory is insufficient

### Performance Notes

- First run will be slower due to model downloading (~20GB)
- Subsequent runs will be faster as the model is cached
- GPU acceleration significantly improves performance
- The model processes images at 1024x1024 resolution by default

## File Structure

```
flux_kontext_lyrics/
├── build_kontext_ds.py                    # Dataset generation script
├── analyze_samples.py                     # Analysis and evaluation script
├── main.py                                # Legacy image editing script
├── requirements.txt                       # Python dependencies
├── kontext_run_small_new/                 # Generated training dataset
│   ├── before/                           # Input images
│   ├── after/                            # Target images with captions
│   ├── test/                             # Test input images
│   ├── test_results/                     # Ground truth test images
│   └── kontext_lora_config.yaml          # Training configuration
├── output_fludyn/                        # Training outputs
│   ├── samples/                          # Generated samples
│   └── analysis/                         # Analysis results
├── ai-toolkit/                           # Ostris AI toolkit (cloned)
└── README.md                             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project uses the FLUX.1-Kontext-dev model which has its own licensing terms. Please review the model's license on Hugging Face before use.