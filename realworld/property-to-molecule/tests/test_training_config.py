import json
import pytest

from sedge_molecule.policy import training_config
from sedge_molecule.train import lr_at


def test_paper_schedule_and_explicit_override_validation(tmp_path):
    config = training_config()
    assert lr_at(0, config, 8700) == pytest.approx(5e-6)
    assert lr_at(199, config, 8700) == pytest.approx(1e-3)
    assert lr_at(200, config, 8700) == pytest.approx(1e-3)
    assert lr_at(8700, config, 8700) == pytest.approx(1e-4)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"epoch": 50}))
    with pytest.raises(ValueError, match="Unknown"):
        training_config(path)
    path.write_text(json.dumps({"peak_lr": float("nan")}))
    with pytest.raises(ValueError, match="Non-finite"):
        training_config(path)
