#!/usr/bin/env python3
"""OpenFold3-JAX: predict structure from sequence.

Usage:
    python predict.py "MQIFVK..."                         # single chain
    python predict.py "CHAIN1/CHAIN2"                     # multi-chain
    python predict.py --fasta input.fasta                  # from FASTA
    python predict.py --query query.json                   # full JSON query
    python predict.py "MQIFVK..." --msa-server             # with MSA search
    python predict.py "MQIFVK..." --model /path/to/model    # custom model path
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jopenfold3.batch import Batch
from jopenfold3.metrics import compute_plddt


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs."""
    entries = []
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            elif line:
                seq_lines.append(line)
    if header is not None:
        entries.append((header, "".join(seq_lines)))
    return entries


def build_query_set(args) -> "InferenceQuerySet":
    """Build InferenceQuerySet from CLI arguments."""
    from jopenfold3._vendor.openfold3.core.data.resources.residues import MoleculeType
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Chain, InferenceQuerySet, Query,
    )

    if args.query:
        return InferenceQuerySet.from_json(args.query)

    # Build chains from sequence(s)
    if args.fasta:
        entries = parse_fasta(Path(args.fasta))
    elif args.sequence:
        segments = args.sequence.split("/")
        entries = [(f"chain_{i}", seg) for i, seg in enumerate(segments)]
    else:
        print("Error: provide a sequence, --fasta, or --query", file=sys.stderr)
        sys.exit(1)

    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chains = []
    for i, (name, seq) in enumerate(entries):
        cid = chain_ids[i] if i < len(chain_ids) else f"chain{i}"
        chains.append(Chain(
            molecule_type=MoleculeType.PROTEIN,
            chain_ids=[cid],
            description=name,
            sequence=seq,
        ))

    query = Query(query_name="prediction", chains=chains)
    return InferenceQuerySet(
        seeds=args.seeds,
        queries={"prediction": query},
    )


# ---------------------------------------------------------------------------
# MSA handling
# ---------------------------------------------------------------------------

def compute_msas(query_set, args) -> "InferenceQuerySet":
    """Optionally run ColabFold MSA search or create dummy MSAs."""
    from jopenfold3._vendor.openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        augment_main_msa_with_query_sequence,
        preprocess_colabfold_msas,
    )

    msa_dir = Path(args.msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)
    settings = MsaComputationSettings(msa_output_directory=msa_dir)

    if args.msa_server:
        print("  Running ColabFold MSA search...")
        query_set = preprocess_colabfold_msas(query_set, settings)
    else:
        query_set = augment_main_msa_with_query_sequence(query_set, settings)

    return query_set


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def featurize(query_set, seed) -> tuple[dict, "AtomArray"]:
    """Run the PyTorch featurization pipeline, return (features_dict, atom_array)."""
    from jopenfold3._vendor.openfold3.core.data.framework.single_datasets.inference import InferenceDataset
    from jopenfold3._vendor.openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_configs import InferenceJobConfig
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings, TemplateSettings,
    )

    config = InferenceJobConfig(
        query_set=query_set,
        seeds=[seed],
        msa=MSASettings(subsample_main=False),
        template=TemplateSettings(take_top_k=True),
        template_preprocessor_settings=TemplatePreprocessorSettings(),
    )
    dataset = InferenceDataset(config)
    features = dataset[0]

    if not features.get("valid_sample", torch.tensor([False])).item():
        print("Error: featurization failed", file=sys.stderr)
        sys.exit(1)

    # Extract atom_array before JAX conversion
    atom_array = features.pop("atom_array")
    # Drop non-feature metadata
    for key in ["query_id", "seed", "repeated_sample", "valid_sample"]:
        features.pop(key, None)

    return features, atom_array


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load the JAX model (auto-downloads and converts on first use)."""
    from jopenfold3.model import OpenFold3

    model_path = Path(args.model).expanduser()
    print(f"  Loading model from {model_path} ...")
    return OpenFold3.load(model_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenFold3-JAX: predict protein structure from sequence",
    )
    parser.add_argument(
        "sequence", nargs="?", default=None,
        help="Protein sequence(s), /-separated for multi-chain",
    )
    parser.add_argument("--fasta", type=str, help="Input FASTA file")
    parser.add_argument("--query", type=str, help="Input JSON query file")
    parser.add_argument(
        "--output", type=str, default="./predictions",
        help="Output directory (default: ./predictions)",
    )
    parser.add_argument(
        "--model", type=str, default="~/.openfold3/jax/of3",
        help="JAX model path (auto-downloads on first use)",
    )
    parser.add_argument(
        "--msa-server", action="store_true",
        help="Use ColabFold MSA server for MSA search",
    )
    parser.add_argument(
        "--msa-dir", type=str, default="/tmp/of3_msas",
        help="MSA cache directory (default: /tmp/of3_msas)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Random seeds (default: 0)",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Diffusion samples per seed (default: 5)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Diffusion sampling steps (default: 200)",
    )
    parser.add_argument(
        "--format", type=str, default="cif", choices=["cif", "pdb"],
        help="Output structure format (default: cif)",
    )
    args = parser.parse_args()

    if not args.sequence and not args.fasta and not args.query:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("OpenFold3-JAX Prediction")
    print("=" * 40)

    # --- 1. Parse input ---
    query_set = build_query_set(args)
    query_name = list(query_set.queries.keys())[0]
    query = query_set.queries[query_name]
    n_chains = len(query.chains)
    total_residues = sum(len(c.sequence) for c in query.chains if c.sequence)
    print(f"Input: {n_chains} chain(s), {total_residues} residues")
    if args.msa_server:
        print("MSA: ColabFold server")
    else:
        print("MSA: query-only (use --msa-server for better accuracy)")

    # --- 2. MSA computation ---
    query_set = compute_msas(query_set, args)

    # --- 3. Load model ---
    print()
    model = load_model(args)

    # --- Run for each seed ---
    confidence_summary = {}

    for seed in args.seeds:
        seed_label = f"seed_{seed}"
        print(f"\n--- Seed {seed} ---")

        # --- 4. Featurize ---
        t0 = time.time()
        print("[1/4] Featurizing...", end="", flush=True)
        features, atom_array = featurize(query_set, seed)
        n_atoms = atom_array.array_length()
        t_feat = time.time() - t0
        print(f"  done ({t_feat:.1f}s, {n_atoms} atoms)")

        # --- 5. Convert to JAX ---
        batch = Batch.from_torch_dict(features)

        # --- 6. Inference ---
        key = jax.random.PRNGKey(seed)

        t0 = time.time()
        print(f"[2/4] Running model ({args.steps} steps, {args.samples} samples)...",
              end="", flush=True)
        with jax.default_matmul_precision("float32"):
            output = model(
                batch,
                num_sampling_steps=args.steps,
                num_samples=args.samples,
                key=key,
            )
        jax.block_until_ready(output.coordinates)
        t_model = time.time() - t0
        print(f"  done ({t_model:.1f}s)")

        # --- 7. Write outputs ---
        from jopenfold3._vendor.openfold3.core.runners.writer import OF3OutputWriter

        t0 = time.time()
        print(f"[3/4] Writing {args.samples} samples...", end="", flush=True)

        seed_dir = output_dir / seed_label
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_confidence = {}

        for s in range(args.samples):
            coords = np.array(output.coordinates[0, s])  # [N_atom, 3]
            plddt = compute_plddt(output.confidence.plddt_logits[0, s])  # [N_atom]
            avg_plddt = float(np.mean(plddt))
            seed_confidence[f"sample_{s + 1}"] = avg_plddt

            out_path = seed_dir / f"sample_{s + 1}.{args.format}"
            OF3OutputWriter.write_structure_prediction(
                atom_array=atom_array.copy(),
                predicted_coords=coords,
                plddt=plddt,
                output_file=out_path,
            )

        t_write = time.time() - t0
        print(f"  done ({t_write:.1f}s)")

        # --- 8. Summary ---
        confidence_summary[seed_label] = seed_confidence
        print()
        print(f"Results in {seed_dir}/")
        for name, plddt_val in seed_confidence.items():
            print(f"  {name}.{args.format}  pLDDT={plddt_val:.1f}")

    # Write confidence summary
    conf_path = output_dir / "confidence.json"
    with open(conf_path, "w") as f:
        json.dump(confidence_summary, f, indent=2)
    print(f"\nConfidence scores: {conf_path}")
    print()


if __name__ == "__main__":
    main()
