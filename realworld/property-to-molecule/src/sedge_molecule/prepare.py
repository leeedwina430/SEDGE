"""Fit the vocabulary/scaler on training data and save scratch initialization."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--custom-data", action="store_true", help="Explicitly mark a new dataset, not the paper split")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    from sedge_molecule import upstream as upstream
    from sedge_molecule.data import Scaler
    from .data import read_train_csv
    from .model import ConditionedCGenerator
    from .io import seed_all, atomic_json, atomic_save, environment
    from .policy import PROTOCOL

    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing asset directory")
    split = read_train_csv(args.train_csv, custom=args.custom_data)
    from rdkit import Chem
    for index, smiles in enumerate(split.smiles):
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid training SMILES at row {index + 2}")
    # Exactly the historical train-only vocabulary construction, including sort order and valence fit.
    tokens, valences = upstream.smiles_list2atoms_list(list(split.smiles), TOKEN2ATOMFEAT={}, VALENCES={})
    valences = upstream.get_max_valence_from_dataset(list(split.smiles), VALENCES=valences)
    vocab = upstream.SpanningTreeVocabulary(tokens, valences, sort=True)
    scaler = Scaler.fit(split.properties)
    seed_all(args.seed)
    model = ConditionedCGenerator(vocab)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_save(dict(vocab=vocab, token2atomfeat=tokens, valences=valences,
                     fit_split="train", fit_rows=split.rows), args.output / "vocabulary.pt")
    (args.output / "scaler.json").write_text(scaler.to_json())
    atomic_save(model.state_dict(), args.output / "initial_model.pt")
    manifest = dict(schema="sedge_ccond_assets_v2", arm="S_STRUCT_CCOND", seed=args.seed,
                    dataset_mode="custom" if args.custom_data else "paper_train",
                    train_rows=split.rows,
                    fit_split="train", property_order=["mol_logp", "tpsa"],
                    model_config=model.cfg.to_dict(), c_conditioner=PROTOCOL["c_conditioner"],
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    environment=environment())
    atomic_json(args.output / "manifest.json", manifest)
    print(f"Prepared {split.rows} training rows; {len(vocab.TOKENS)} tokens; {manifest['parameter_count']:,} parameters")


if __name__ == "__main__":
    main()
