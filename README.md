# OpenFold3-JAX

JAX/Equinox translation of [OpenFold3](https://github.com/aqlaboratory/openfold3) (AlphaFold 3).
Featurization uses the upstream PyTorch pipeline; the model itself — recycling,
diffusion sampling, and confidence heads — runs entirely in JAX.

## Installation

```bash
git clone https://github.com/escalante-bio/jopenfold3.git
cd jopenfold3
uv sync
```

Requires Python 3.12+, NVIDIA GPU with CUDA 12, and
[uv](https://github.com/astral-sh/uv).

## Usage

```bash
# Predict (auto-downloads and converts weights on first run)
python predict.py "MQIFVK..."
python predict.py "SEQA/SEQB"              # multi-chain
python predict.py --fasta input.fasta
python predict.py "MQIFVK..." --msa-server  # ColabFold MSA search
```

Key options: `--steps` (default 200), `--samples` (default 5), `--seeds`,
`--format {cif,pdb}`, `--output`.

## As a dependency

```bash
uv add git+https://github.com/escalante-bio/jopenfold3.git
```

```python
from jopenfold3.model import OpenFold3

model = OpenFold3.load()  # auto-downloads on first call
```

## License

MIT
