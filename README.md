# Stable Diffusion with Promptable Image Generation on M1 Mac's using DeepCache

#### Authors: Yiming Zheng (yiming.zheng@mail.utoronto.ca), Xinran Zhang (xinran.zhang@mail.utoronto.ca), Barry Xue (zexin.xue@mail.utoronto.ca)

## Description

This repository implements a version of the DeepCache algorithm in MLX.

We based our implementation on these two official repositories from Apple and the DeepCache author.

- DeepCache: https://github.com/horseee/DeepCache/tree/master.
- MLX Stable Diffusion Example: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion.

MLX is a DL framework developed by Apple to accelerate model inference on M1 architecture through the METAL API. An advantage MLX has over other frameworks is that MLX supports memory sharing between the M1 CPU and GPU, and the code evaluates lazily.

Our goal is to use this as a proof-of-concept to demonstrate how MLX can be combined with DeepCache to enable fast, on-device inference.

## Running our Code

### Structure

We proved the original MLX Stable Diffusion demonstration in `mlx_stable_diffusion`, and our implementation with DeepCache based on the official demo in `mlx_deepcache`.

Both directories follow the same structure and can be run with the same set of commands.

### Example

To run the official text-to-image example with a prompt run the following command:

```shell
cd mlx_stable_diffusion
python txt2image.py "A photo of an astronaut riding a horse on Mars." --n_images 4 --n_rows 2
```

To run the same example with DeepCache, run these commands instead:

```shell
cd mlx_deepcache
python txt2image.py "A photo of an astronaut riding a horse on Mars." --n_images 4 --n_rows 2 --cache_interval 3
```

For further information on running another type of experiment, please refer to either `mlx_stable_diffusion/README.md` or `mlx_deepcache/txt2image.py`.

> Currently, only Stable Diffusion is supported (run this model with the `--model sd` flag) for the DeepCache version, and SDXL Turbo is still under development.
