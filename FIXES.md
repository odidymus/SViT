# Why the snnTorch port trained badly, and what changed

Checked against a SpikingJelly reference of upstream Spikingformer, using the same
weights in float64. After the fix the forward pass, gradients (global relative error
about 1e-15) and BatchNorm running stats all match exactly. The old port did not:
logits were off by 0.24, gradients by up to 20x, and BN running stats by 0.24.

## 1. BatchNorm saw one timestep at a time (main bug)
The port looped over time *outside* the network, so every BatchNorm ran T separate
times per batch on B samples each, instead of once on T*B samples.
- In training, each timestep was normalized with its own statistics. That is a
  different model from upstream.
- The running stats were updated T times per batch and mostly reflected the last
  timesteps, so eval mode (running stats) did not match what training saw. This
  costs validation accuracy.

Fix: layers now run one after another over the whole sequence, the way upstream
does. Each `MultiStepLIF` loops the native snnTorch `LIF` over T, and conv/BN layers
fold T into the batch (`seq_apply`). The neurons are still snnTorch single-step
neurons. `forward_step()` still works for streaming inference and in eval mode
gives the same result as `forward()`.

## 2. The spike condition was `>` instead of `>=`
SpikingJelly fires when `v >= threshold`. The port's surrogate used `v > threshold`.
The attention-output LIF (threshold 0.5) receives multiples of 0.125, so values
landing exactly on the threshold are common, and the port dropped those
attention spikes.

## 3. Smaller fixes
- The attention product now uses upstream's order, `Q @ (K^T @ V)`, which is
  cheaper for N > d.
- Membrane tensors are released after each forward. They are registered buffers,
  so otherwise DDP broadcasts them, ModelEma deepcopy breaks, and memory stays pinned.
- The `Spikingformer` factory ignores extra kwargs that newer timm versions inject.

## train.py
- Checkpoints go to `models/<run-name>/` (`model_best.pth.tar`, `last.pth.tar`,
  `checkpoint-*.pth.tar`, `summary.csv`, `args.yaml`). The newest best is also
  copied to `models/best_model.pth.tar`.
- The per-batch NaN/grad diagnostic now runs only with `--debug-grads`. It
  synced the GPU once per parameter on every step.
- `--initial-checkpoint` now works (it was ignored before). Added `--dataset-download`.
- The config path is resolved relative to the repo root as well as the current
  directory. `src/` is added to `sys.path`.
- `convert_splitbn_model` imports on both old and new timm.

## Config
`configs/cifar10_upstream.yml` holds the upstream CIFAR-10 recipe (AdamW 5e-4,
wd 0.06, cosine, 400 epochs, mixup 0.5, RandAugment, `scale: [1, 1]`). If your own
config leaves `scale` unset, timm defaults to random-resized crops with scale
0.08-1.0, which is very destructive on 32x32 images.

    python scripts/train.py -c configs/cifar10_upstream.yml --dataset-download
