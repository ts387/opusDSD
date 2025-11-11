# OPUS-DSD on Apple Silicon (M-Series Macs)

This document describes how to run OPUS-DSD on Apple Silicon Macs (M1, M2, M3, etc.) with Metal Performance Shaders (MPS) GPU acceleration.

## Overview

OPUS-DSD has been adapted to support Apple Silicon Macs through PyTorch's MPS backend. This allows GPU-accelerated training and inference on M-Series chips without requiring CUDA.

## Key Differences from CUDA Version

1. **Single GPU Only**: MPS does not support multi-GPU training via DataParallel. The `--multigpu` flag will be automatically disabled on Apple Silicon.

2. **MPS Backend**: Instead of CUDA, PyTorch uses Metal Performance Shaders for GPU acceleration.

3. **Batch Normalization**: Synchronized Batch Normalization (used in multi-GPU CUDA training) automatically falls back to standard Batch Normalization on MPS.

## Installation

### Prerequisites
- macOS 12.3 or later
- Apple Silicon Mac (M1, M2, M3, or later)
- Conda or Miniconda

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alncat/opusDSD.git
   cd opusDSD
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f environment_macos_arm64.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate opusdsd-mps
   ```

4. **Install OPUS-DSD**:
   ```bash
   pip install -e .
   ```

5. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

   You should see output indicating that MPS is available.

## Usage

### Training

The training commands remain the same as the CUDA version, but you should:

1. **Omit or ignore the `--multigpu` flag**: Multi-GPU training is not supported on MPS. The flag will be automatically disabled with a warning.

2. **Adjust batch size**: Since you're limited to a single GPU, you may need to reduce the batch size compared to multi-GPU CUDA setups.

Example training command:
```bash
dsd train_multi /path/to/particles.mrcs \
    --ctf /path/to/ctf.pkl \
    --poses /path/to/poses.pkl \
    -n 50 \
    -b 8 \
    --zdim 12 \
    --lr 1e-4 \
    -o /path/to/output \
    -r /path/to/mask.mrc \
    --beta cos \
    --beta-control 2.0
```

**Note**: Do NOT use `--multigpu` or `--num-gpus` flags, or if you do, they will be ignored.

### Memory Considerations

Apple Silicon uses unified memory architecture where GPU and CPU share the same memory pool. Monitor your memory usage:

```bash
# During training, you can monitor memory in another terminal
top -l 1 | grep PhysMem
```

If you encounter out-of-memory errors:
- Reduce batch size (`-b` parameter)
- Reduce image dimensions
- Reduce model complexity (`--zdim`)

### Performance Expectations

- **M1/M2/M3 Max/Pro/Ultra**: Excellent performance for single-GPU training, comparable to mid-range NVIDIA GPUs
- **M1/M2/M3 Base**: Good performance but may require smaller batch sizes due to memory constraints

## Known Limitations

1. **No Multi-GPU Support**: MPS backend doesn't support DataParallel or DistributedDataParallel
2. **No Mixed Precision (APEX)**: APEX (Automatic Mixed Precision) is CUDA-specific and won't work
3. **Synchronized BatchNorm**: Falls back to standard BatchNorm (functionally equivalent for single GPU)

## Troubleshooting

### MPS Not Available

If `torch.backends.mps.is_available()` returns `False`:

1. **Check macOS version**: Requires macOS 12.3+
   ```bash
   sw_vers
   ```

2. **Update PyTorch**: Ensure you have PyTorch 2.0 or later
   ```bash
   conda list pytorch
   ```

3. **Check for arm64**: Verify you're using native arm64 packages
   ```bash
   python -c "import platform; print(platform.machine())"
   ```
   Should output `arm64`

### Out of Memory Errors

If you encounter memory errors:

```bash
# Reduce batch size
dsd train_multi ... -b 4  # or even -b 2

# Or use a smaller image size
dsd train_multi ... --downfrac 0.5
```

### Slow Performance

If training is slower than expected:

1. **Verify MPS is being used**:
   Check the log output at the start of training. You should see:
   ```
   Using device: mps
   ```

2. **Check for CPU fallback**:
   If you see `Using device: cpu`, MPS may not be properly configured.

3. **Monitor GPU usage**:
   ```bash
   sudo powermetrics --samplers gpu_power -i 1000 -n 1
   ```

### Package Conflicts

If conda has trouble resolving dependencies:

```bash
# Use mamba for faster dependency resolution
conda install -c conda-forge mamba
mamba env create -f environment_macos_arm64.yml
```

## Compatibility Notes

- **Rosetta 2**: Do not run under Rosetta 2 emulation. Use native arm64 Python and packages.
- **Intel Macs**: This environment is specifically for Apple Silicon. Intel Macs should use the standard CUDA or CPU environment.

## Performance Tips

1. **Use native arm64 packages**: Avoid x86_64 packages that run through Rosetta
2. **Optimize batch size**: Experiment with batch sizes to find the sweet spot for your Mac's memory
3. **Monitor memory pressure**: Use Activity Monitor to watch memory usage
4. **Close other applications**: Free up memory for training

## Reporting Issues

When reporting issues specific to Apple Silicon:

1. Include your Mac model (M1/M2/M3, Pro/Max/Ultra)
2. Include macOS version (`sw_vers`)
3. Include PyTorch version and MPS availability
4. Include the full error message and traceback
5. Specify the command you ran

## Additional Resources

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [OPUS-DSD Main Documentation](README.md)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)

## Credits

Apple Silicon support added as part of the OPUS-DSD port to Metal Performance Shaders.
Original OPUS-DSD by the Ma Lab at Fudan University.
