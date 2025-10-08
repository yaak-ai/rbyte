<p align="center">
 <a href="https://www.yaak.ai/open-source/dev-tools">
  <img alt="banner" src="https://github.com/user-attachments/assets/707ab3ae-73d5-459f-82c5-888323673adb">
 </a>
</p>

<p align="center">
 <img src="https://github.com/yaak-ai/rbyte/actions/workflows/ci.yaml/badge.svg">
 <img src="https://img.shields.io/github/license/yaak-ai/rbyte.svg?color=green"></a>
</p>

`rbyte` implements a [PyTorch](https://pytorch.org) [`Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) with [`TensorClass`](https://pytorch.org/tensordict/main/reference/tensorclass.html)-based samples built from multimodal data.

## Installation

```bash
uv add rbyte [--extra <EXTRA>]
```

See `pyproject.toml` for available extras.

## Examples

```bash
git clone https://github.com/yaak-ai/rbyte && cd rbyte
nix develop # alternatively, install `uv`, `just`, `ytt`
just notebook examples/nuscenes.ipynb
```

## Development

```bash
git clone https://github.com/yaak-ai/rbyte && cd rbyte
nix develop
just setup
```
