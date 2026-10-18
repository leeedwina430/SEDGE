import json

import pytest

from sedge_molecule.requests import main, read_requests


def test_requests_are_frozen_sorted_unique_and_refuse_overwrite(tmp_path):
    conditions = tmp_path / "conditions.csv"
    conditions.write_text("condition_id,mol_logp,tpsa\nb,6,110\na,5.5,120\n")
    output = tmp_path / "requests.jsonl"
    args = ["--conditions-csv", str(conditions), "--panel", "test_ood", "--seeds", "2", "0",
            "--samples-per-condition", "3", "--output", str(output)]
    main(args)
    rows = read_requests(output)
    assert len(rows) == len({r["request_id"] for r in rows}) == 12
    assert [(r["sampling_seed"], r["condition_id"], r["draw_index"]) for r in rows] == sorted(
        (seed, condition, draw) for seed in (0, 2) for condition in ("a", "b") for draw in range(3))
    with pytest.raises(ValueError, match="overwrite"):
        main(args)
    rows[1]["request_id"] = rows[0]["request_id"]
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Duplicate request_id"):
        read_requests(output)


def test_nonfinite_conditions_never_publish_a_request_file(tmp_path):
    conditions = tmp_path / "conditions.csv"
    conditions.write_text("condition_id,mol_logp,tpsa\na,nan,120\n")
    output = tmp_path / "requests.jsonl"
    with pytest.raises(ValueError):
        main(["--conditions-csv", str(conditions), "--panel", "test_ood", "--output", str(output)])
    assert not output.exists()


def test_id_panel_selects_conditions_in_lexicographic_order(tmp_path):
    import numpy as np
    from sedge_molecule.data import Split
    from sedge_molecule._request_inventory import build_id_diagnostic

    ids = tuple(f"condition-{i}" for i in range(140))
    split = Split("val_id", tmp_path / "val_id.csv", 140, ids,
                  tuple(["CCO"] * 140), np.array([[i / 10, 10 + i] for i in range(140)]), ids)
    rows, metadata = build_id_diagnostic(split)
    rows.sort(key=lambda r: (r.panel, r.sampling_seed, r.condition_id, r.draw_index, r.request_id))
    assert len(rows) == 1280
    assert len({r.condition_id for r in rows}) == metadata["unique_conditions"] == 128
    assert all(r.observed == (True, True) and "|m11|s0|" in r.request_id for r in rows)

    assert {r.condition_id for r in rows} == set(sorted(ids)[:128])
