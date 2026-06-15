# EcoDPL: Evolving Compact Dual Prompts for Continual De-Raining

This repository contains the release implementation for **Prompting Rain Off: Evolving Compact Dual Prompts for Continual De-Raining**.

The release training path is centered on:

- `net/ecodpl_promptir.py`: EcoDPL model with image prompts, feature prompts, P-Fuser, frequency tables, and Grad-Tuner.
- `train_ecodpl_release.py`: continual deraining training over Rain800, Rain100H, and Rain100L.
- `eval_ecodpl_release.py`: tiled full-image evaluation for large test images.
- `utils/derain_release.py`: dataset loading, metrics, padding, and tiled inference helpers.

## Environment

The code is tested with Python 3.8, PyTorch 2.x, CUDA, OpenCV, h5py, scikit-image, torchvision, einops, and tqdm.

On Titan8, the working environment used during verification is:

```bash
/home/liumh/anaconda3/envs/quadprior/bin/python
```

## Data Layout

Set `--data-root` to a directory with the following layout:

```text
Image-deraining/
  Rain800/
    train_input.h5
    train_target.h5
    inputTest/
    targetTest/
  RainTrainH/
    train_input.h5
    train_target.h5
  RainTrainL/
    train_input.h5
    train_target.h5
  RainTestH/
    rain/
    norain/
  RainTestL/
    rain/
    norain/
```

The default Titan8 path is `/mnt/netdisk/liumh/workspace/Image-deraining`.

## Train

Rain800 -> Rain100H:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_ecodpl_release.py \
  --data-root /path/to/Image-deraining \
  --tasks Rain800 Rain100H \
  --epochs-per-task 50 \
  --batch-size 18 \
  --patch-size 100 \
  --num-workers 4 \
  --importance-batches 50 \
  --importance-batch-size 4 \
  --tile-size 384 \
  --tile-overlap 32 \
  --amp \
  --no-perceptual \
  --output-dir runs/ecodpl_r800_r100h
```

Remove `--no-perceptual` to use the paper-style perceptual term with `--perceptual-weight 0.04`.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python eval_ecodpl_release.py \
  --data-root /path/to/Image-deraining \
  --task Rain800 \
  --checkpoint runs/ecodpl_r800_r100h/best_Rain800.pth \
  --tile-size 384 \
  --tile-overlap 32
```

Use `--task Rain100H` or `--task Rain100L` for the corresponding test sets.

## Verification Status

Training and full-image tiled evaluation are currently being verified on Titan8. The verified checkpoint metrics will be added here before public release.

## License

This code builds on the PromptIR/Restormer-style backbone already present in the repository. See `LICENSE.md` for the non-commercial academic license terms.
