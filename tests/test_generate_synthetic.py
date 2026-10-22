import pytest
import yaml

from plate_detectors.dataset.generate_synthetic import (
    class_mapping,
    remap_label_file,
    split_into,
    to_gasp_name,
    write_data_yaml,
)
from plate_detectors.paths import CONFIGS_DIR

GASP_NAMES = {0: "observacion", 1: "science", 2: "lamp"}


def test_mapping_for_repo_configs():
    observation = yaml.safe_load((CONFIGS_DIR / "observation.yaml").read_text())["names"]
    components = yaml.safe_load((CONFIGS_DIR / "components.yaml").read_text())["names"]
    assert class_mapping(GASP_NAMES, observation) == {0: 0}
    assert class_mapping(GASP_NAMES, components) == {1: 1, 2: 0}  # science=1, lamp 2 -> 0


def test_target_names_translate_to_gasp_names():
    assert [to_gasp_name(n) for n in ("observation", "lamp", "science")] == ["observacion", "lamp", "science"]


def test_remap_label_file_keeps_coordinates(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")
    remap_label_file(label, {1: 1, 2: 0})
    assert label.read_text() == "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6"


def test_remap_rejects_unknown_class(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4")
    with pytest.raises(ValueError):
        remap_label_file(label, {1: 1, 2: 0})


def test_split_into_moves_pairs_and_writes_yaml(tmp_path):
    raw, out = tmp_path / "out" / "_gasp", tmp_path / "out"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    for i in range(10):
        (raw / "images" / f"{i}.jpg").write_bytes(b"x")
        (raw / "labels" / f"{i}.txt").write_text(str(i))

    counts = split_into(raw, out, val_fraction=0.2, seed=0)

    assert counts == {"train": 8, "val": 2}
    assert not any((raw / "images").iterdir())
    for split in ("train", "val"):
        for img in (out / "images" / split).iterdir():
            assert (out / "labels" / split / f"{img.stem}.txt").read_text() == img.stem
    data = yaml.safe_load(write_data_yaml(out, {0: "observation"}).read_text())
    assert data == {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                    "names": {0: "observation"}}
