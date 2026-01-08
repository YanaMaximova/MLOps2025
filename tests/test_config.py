from pathlib import Path
import yaml


config_path = Path(__file__).parent.parent / "config.yaml"

def test_config_structure_and_types():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for section in ["data", "model", "train", "inference", "save"]:
        assert section in config, f"Missing section: {section}"

    assert isinstance(config["data"]["train_img_dir"], str)
    assert isinstance(config["data"]["train_gt_path"], str)
    assert 0.0 < config["data"]["fraction"] < 1.0

    assert config["model"]["backbone"] in ["efficientnet_b3", "mobilenet_v2"]
    assert config["model"]["num_classes"] == 50
    assert isinstance(config["model"]["pretrained"], bool)
    assert isinstance(config["model"]["training_layers"], int)

    assert isinstance(config["train"]["batch_size"], int) and config["train"]["batch_size"] > 0
    assert isinstance(config["train"]["lr"], (int, float)) and config["train"]["lr"] > 0
    assert isinstance(config["train"]["max_epochs"], int) and config["train"]["max_epochs"] > 0

    assert isinstance(config["inference"]["batch_size"], int) and config["inference"]["batch_size"] > 0
    assert config["inference"]["device"] in ["auto", "cpu", "mps", "cuda"]