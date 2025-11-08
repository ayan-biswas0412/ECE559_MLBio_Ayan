# Advanced Personalized Arrhythmia Detection Framework with Complex Deep Learning Models

## Overview

This project implements a sophisticated, production-ready framework for detecting cardiac arrhythmias using advanced deep learning models. The system combines multiple personalized neural network architectures (CNN, BiLSTM, and attention-based models) with an intelligent orchestrator that dynamically selects the optimal model for each patient based on their unique cardiac patterns.

The framework leverages the **MIT-BIH Arrhythmia Database**, a benchmark dataset containing 30 minutes of continuous ECG recordings from 47 subjects with various cardiac conditions. This enables the detection of five major arrhythmia types: Normal (N), Atrial Premature Contraction (A), Left Ventricular Premature Contraction (L), Right Ventricular Premature Contraction (R), and Ventricular Fibrillation (V).

### Key Features

- **Multi-Algorithm Ensemble**: Three specialized deep learning models tailored for different aspects of ECG signal analysis
- **Intelligent Orchestrator**: Adaptive model selection mechanism that optimizes performance on a per-patient basis
- **Attention Mechanisms**: Transformer-inspired attention layers for capturing long-range dependencies in cardiac signals
- **Comprehensive Visualization**: Detailed performance metrics, confusion matrices, ROC curves, and TSNE embeddings
- **One-Click Setup**: Single installation cell handles all dependencies and data downloads
- **Production-Ready Code**: Full reproducibility with proper data preprocessing, normalization, and validation strategies

---

## Quick Start (3 Simple Steps)

### Step 1: Open the Notebook
```bash
jupyter notebook arrhythmia_predict.ipynb
```

### Step 2: Run the Installation Cell (Only Once!)
Execute the **first cell** in the notebook marked as `[Single Time] Installation of Required Packages and Datasets`. This single cell will:
- Download the entire MIT-BIH Arrhythmia Database (~90 MB)
- Install all required Python packages

```python
# =====================================================
# [Single Time] Installation of Required Packages and Datasets
# =====================================================

# Install the MIT-BIH Arrhythmia Database dataset
!wget -r -N -c -np -P mit-bih-dataset https://physionet.org/files/mitdb/1.0.0/

# Install required packages if not already installed
!pip install --quiet --upgrade pip
!pip install --quiet numpy pandas matplotlib seaborn tensorflow scikit-learn wfdb
```

**Just press Shift + Enter to run this cell once. That's it!**

### Step 3: Run All Cells
After the installation cell completes successfully, run all remaining cells sequentially:
- **Cell → Run All** (menu option), or
- **Shift + Enter** on each cell individually

**Expected total runtime**: 15-30 minutes (depending on your hardware)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **CPU**: Dual-core processor (2 GHz or higher)
- **RAM**: 8 GB
- **Disk Space**: 2 GB (including dataset and model checkpoints)
- **OS**: Windows, macOS, or Linux
- **Jupyter Notebook**: Pre-installed (or will be installed via the setup cell)

### Recommended Requirements
- **CPU**: Quad-core processor (3 GHz or higher)
- **GPU**: NVIDIA GPU with CUDA support (10-50x faster training)
- **RAM**: 16 GB or higher
- **Disk Space**: 5 GB

---

## What the Installation Cell Does

When you run the first cell, it automatically:

1. **Downloads the MIT-BIH Dataset**
   - 48 ECG records (30 minutes each)
   - Stored in a `mit-bih-dataset/` folder
   - Uses wget with resume capability (safe to re-run if interrupted)

2. **Installs All Dependencies**
   - numpy, pandas, matplotlib, seaborn (data processing & visualization)
   - tensorflow (deep learning framework)
   - scikit-learn (machine learning utilities)
   - wfdb (ECG data format reader)

**No additional setup or terminal commands needed!**

---

## Project Structure

```
arrhythmia_predict.ipynb              # Main notebook (all code & analysis)
├── Cell 1: [Single Time] Installation Cell
├── Cell 2: Imports and Configuration
├── Cell 3-5: Data Loading and Preprocessing
├── Cell 6-10: Model Architecture Definitions
├── Cell 11-15: Model Training
├── Cell 16-20: Performance Evaluation
└── Cell 21-25: Visualizations and Analysis

mit-bih-dataset/                      # ECG data directory (auto-created)
├── 100.dat, 100.hea, 100.atr
├── 101.dat, 101.hea, 101.atr
├── ... (48 additional records)
└── RECORDS
```

---

## How to Use

### For First-Time Users
1. Open `arrhythmia_predict.ipynb` in Jupyter
2. Run the installation cell (the very first one marked with [Single Time])
3. Wait for completion (~5-10 minutes for downloads)
4. Run all remaining cells in order

### For Subsequent Runs
- Simply open the notebook and run all cells
- The installation cell will skip downloads if files already exist (due to `wget -N` flag)
- Training will begin immediately

### Making Predictions
The notebook will:
1. Load the MIT-BIH dataset automatically
2. Preprocess and normalize ECG signals
3. Train three deep learning models (CNN, BiLSTM, Attention-based)
4. Use the orchestrator to select the best model per patient
5. Generate comprehensive performance visualizations

---

## What You'll Learn

The notebook demonstrates:

- **ECG Signal Processing**: Loading, preprocessing, and normalizing cardiac data
- **Deep Learning Architecture Design**: Building CNN, BiLSTM, and Attention-based models
- **Ensemble Methods**: Combining multiple models for robust predictions
- **Performance Metrics**: Computing accuracy, precision, recall, F1-score, and AUC-ROC
- **Advanced Visualization**: Confusion matrices, ROC curves, TSNE embeddings
- **Medical AI**: Real-world application of deep learning in cardiac arrhythmia detection

---

## Dataset Information

### MIT-BIH Arrhythmia Database

**Source**: PhysioNet (https://physionet.org/content/mitdb/1.0.0/)

**Dataset Characteristics**:
- 48 half-hour ECG records from 47 subjects
- Sampling rate: 360 Hz
- Duration: 30 minutes per record
- Resolution: 11-bit, ±5 mV range
- Automatically downloaded when you run the installation cell

**Arrhythmia Classes**:
- **N**: Normal beat (approximately 75% of beats)
- **A**: Atrial premature contraction
- **L**: Left ventricular premature contraction
- **R**: Right ventricular premature contraction
- **V**: Ventricular fibrillation/flutter


## Troubleshooting

### Problem: Installation cell takes a long time
**Solution**: This is normal for the first run. The dataset is ~90 MB and model training takes 15-30 minutes. Subsequent runs are faster as data is cached.

### Problem: "wget: command not found"
**Solution**: The `!` prefix makes it run in the notebook environment, not your terminal. If issues persist:
- Windows users: May need to install Git Bash or use Windows Subsystem for Linux (WSL)
- macOS/Linux: wget should be available by default

### Problem: "ModuleNotFoundError" after running installation
**Solution**: The installation cell should handle this automatically. If errors persist:
1. Restart the kernel (Kernel → Restart)
2. Re-run the installation cell
3. Run remaining cells

### Problem: Out of memory error during training
**Solution**: 
- Reduce batch size (edit the notebook variable from 32 to 16)
- Reduce number of epochs
- Use a machine with more RAM (8 GB minimum, 16 GB recommended)
- Enable GPU acceleration if available

### Problem: Dataset folder is empty
**Solution**: 
- Re-run the installation cell
- Check internet connection during download
- Verify the `mit-bih-dataset/` folder was created in your working directory

### Problem: TensorFlow GPU issues
**Solution**: The notebook works with CPU by default. For GPU acceleration:
```python
# Add this to see GPU info:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Advanced Customization

### Adjust Training Parameters
Modify these variables in the notebook for faster/better results:
- `epochs`: Number of training iterations (default: 50)
- `batch_size`: Samples per batch (default: 32)
- `learning_rate`: Optimization step size (default: 0.001)

### Change Model Architecture
Edit the model definitions to add/remove layers or adjust layer sizes for your specific use case.

### Use Your Own ECG Data
Replace the dataset loading section with your data source while maintaining the same preprocessing pipeline.

---

## Citation & References

If you use this framework in your research, please cite:

```bibtex
@dataset{moody2001mit,
  title={MIT-BIH Arrhythmia Database},
  author={Moody, GB and Mark, RG},
  year={1992},
  publisher={PhysioNet},
  url={https://physionet.org/content/mitdb/1.0.0/}
}
```

### Key Papers
- Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet"
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Vaswani, A., et al. (2017). "Attention Is All You Need"

---

## FAQs

**Q: Do I need to install anything before opening the notebook?**
A: No! Just have Python 3.8+ and Jupyter Notebook installed. The notebook's installation cell handles everything else.

**Q: Can I interrupt the installation cell and resume later?**
A: Yes! The wget command with `-c` flag allows resuming interrupted downloads. Just re-run the cell.

**Q: How much disk space do I need?**
A: About 2 GB for the dataset and trained models. Minimum 5 GB recommended.

**Q: Will this work on my laptop?**
A: Yes, if it meets minimum requirements (8 GB RAM, Python 3.8+). Training will be slower without GPU, but it will work.

**Q: Can I use this for production deployment?**
A: Yes! The models are trained using TensorFlow and can be exported for deployment in medical devices or cloud services.

## Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review inline comments in the notebook cells
3. Refer to official documentation:
   - TensorFlow: https://tensorflow.org
   - Scikit-learn: https://scikit-learn.org
   - PhysioNet: https://physionet.org

---

**Ready to get started? Just open `arrhythmia_predict.ipynb` and run the first cell!** 🎯📊❤️
