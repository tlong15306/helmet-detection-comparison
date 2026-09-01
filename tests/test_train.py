"""Kiểm thử pipeline train không cần dataset thật hay tải pretrained weights."""

from pathlib import Path

import pytest
import torch
from torch import nn

from src.train import (
    build_detector_from_config,
    build_optimizer,
    build_scheduler,
    configure_trainable_parameters,
    load_resume_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_config,
    image_sampling_weights,
    _checkpoint_payload,
    validation_regressed,
    VietnamPrioritySampler,
    FocusedVietnamSampler,
)
from src import train as train_module
from src.utils import load_config


class TinyDetector(nn.Module):
    """Detector giả lập đúng giao diện train/eval của Torchvision."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss_classifier": self.scale * images[0].mean()}
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
            for _ in images
        ]


def _config() -> dict:
    return {
        "project": {"seed": 42},
        "model": {"name": "fasterrcnn_resnet50_fpn_v2", "num_classes": 2},
        "data": {
            "image_root": "data/images",
            "train_annotations": "data/train.json",
            "val_annotations": "data/val.json",
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "scheduler": "step_lr",
            "scheduler_step_size": 1,
            "scheduler_gamma": 0.1,
        },
        "runtime": {"num_workers": 0},
        "output": {"best_checkpoint": "best.pth", "last_checkpoint": "last.pth"},
        "classes": {0: "background", 1: "NoHelmet"},
    }


def test_train_one_epoch_updates_tiny_detector_without_gpu():
    model = TinyDetector()
    optimizer = build_optimizer(model, _config()["training"])
    before = model.scale.detach().item()
    loader = [
        (
            (torch.ones((3, 4, 4)),),
            ({"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]), "labels": torch.tensor([1])},),
        )
    ]
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        scaler,
        use_amp=False,
    )

    assert result["train_batches"] == 1.0
    assert result["train_loss"] == 1.0
    assert model.scale.detach().item() < before


def test_checkpoint_round_trip_restores_optimizer_and_epoch(tmp_path: Path):
    config = _config()
    model = TinyDetector()
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    checkpoint = tmp_path / "last.pth"
    save_checkpoint(
        checkpoint,
        _checkpoint_payload(model, optimizer, scheduler, 3, config, {"map_50_95": 0.2}),
    )

    resumed_model = TinyDetector()
    resumed_optimizer = build_optimizer(resumed_model, config["training"])
    resumed_scheduler = build_scheduler(resumed_optimizer, config["training"])
    next_epoch = load_resume_checkpoint(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        torch.device("cpu"),
        "fasterrcnn_resnet50_fpn_v2",
    )

    assert next_epoch == 4
    assert resumed_model.scale.item() == model.scale.item()


def test_resume_rejects_smoke_checkpoint(tmp_path: Path):
    config = _config()
    model = TinyDetector()
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    checkpoint = tmp_path / "smoke.pth"
    save_checkpoint(
        checkpoint,
        _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            1,
            config,
            {"map_50_95": 0.0},
            smoke_test=True,
            run_type="smoke",
        ),
    )
    resumed_model = TinyDetector()
    resumed_optimizer = build_optimizer(resumed_model, config["training"])
    resumed_scheduler = build_scheduler(resumed_optimizer, config["training"])

    with pytest.raises(ValueError, match="smoke test"):
        load_resume_checkpoint(
            checkpoint,
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            torch.device("cpu"),
            "fasterrcnn_resnet50_fpn_v2",
            expected_run_type="pilot",
        )


def test_validate_config_rejects_class_count_mismatch():
    config = _config()
    config["model"]["num_classes"] = 3
    try:
        validate_config(config)
    except ValueError as error:
        assert "num_classes" in str(error)
    else:
        raise AssertionError("Expected config validation to reject class mismatch")


def test_build_detector_uses_weights_from_config(monkeypatch):
    config = _config()
    config["image"] = {"min_size": 512, "max_size": 768}
    config["model"]["weights"] = "NONE"
    calls = {}

    def fake_build_model(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return TinyDetector()

    monkeypatch.setattr(train_module, "build_model", fake_build_model)
    model = build_detector_from_config(config)

    assert isinstance(model, TinyDetector)
    assert calls["kwargs"]["weights"] == "NONE"
    assert calls["kwargs"]["min_size"] == 512


def test_image_sampling_weights_prioritize_nohelmet_and_small_heads():
    class Dataset:
        images = [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
        ]
        annotations_by_image = {
            1: [{"category_id": 1, "bbox": [0, 0, 50, 50], "area": 2500}],
            2: [{"category_id": 2, "bbox": [0, 0, 5, 5], "area": 25}],
        }

    weights = image_sampling_weights(
        Dataset(),
        class_weights={2: 1.5},
        small_object_ratio=0.005,
        small_object_boost=0.4,
    )

    assert weights.tolist() == pytest.approx([1.0, 1.9])


def test_vietnam_priority_sampler_keeps_all_vietnam_images_without_duplication():
    images = [
        *[{"source_dataset": "edgevision_train"} for _ in range(6)],
        *[{"source_dataset": "vietnam_reviewed_train"} for _ in range(6)],
    ]
    sampler = VietnamPrioritySampler(
        images,
        vietnam_source_dataset="vietnam_reviewed_train",
        vietnam_ratio=0.60,
        seed=42,
    )

    indices = list(sampler)
    vietnam_indices = [index for index in indices if index >= 6]
    original_indices = [index for index in indices if index < 6]

    assert len(indices) == 10
    assert sorted(vietnam_indices) == list(range(6, 12))
    assert len(original_indices) == len(set(original_indices)) == 4


def test_focused_vietnam_sampler_keeps_only_target_class_images_without_duplication():
    images = [
        *[{"id": index, "source_dataset": "edgevision_train"} for index in range(6)],
        *[
            {"id": index, "source_dataset": "vietnam_reviewed_train"}
            for index in range(6, 12)
        ],
    ]
    annotations_by_image = {
        6: [{"category_id": 2}],
        7: [{"category_id": 3}],
        8: [{"category_id": 2}],
        9: [{"category_id": 1}],
        10: [{"category_id": 2}],
        11: [{"category_id": 3}],
    }
    sampler = FocusedVietnamSampler(
        images,
        annotations_by_image,
        vietnam_source_dataset="vietnam_reviewed_train",
        focus_category_id=2,
        vietnam_ratio=0.5,
        seed=42,
    )

    indices = list(sampler)
    vietnam_indices = [index for index in indices if index >= 6]
    original_indices = [index for index in indices if index < 6]

    assert sorted(vietnam_indices) == [6, 8, 10]
    assert len(original_indices) == len(set(original_indices)) == 3


def test_configure_trainable_parameters_freezes_only_backbone_body():
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.body = nn.Linear(2, 2)
            self.backbone.fpn = nn.Linear(2, 2)
            self.head = nn.Linear(2, 1)

    model = Detector()
    counts = configure_trainable_parameters(model, {"freeze_backbone_body": True})

    assert all(not parameter.requires_grad for parameter in model.backbone.body.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.fpn.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert counts["frozen"] > 0
    assert counts["trainable"] > 0


def test_validation_regressed_protects_nohelmet_even_if_map_is_stable():
    previous = {
        "map_50_95": 0.60,
        "per_class": {"NoHelmet": {"ap_50_95": 0.50}},
    }
    current = {
        "map_50_95": 0.60,
        "per_class": {"NoHelmet": {"ap_50_95": 0.49}},
    }
    assert validation_regressed(
        previous,
        current,
        primary_metric="map_50_95",
        protected_classes=["NoHelmet"],
    )


@pytest.mark.parametrize(
    ("config_path", "model_name", "output_directory"),
    [
        ("configs/pilot_faster_rcnn.yaml", "fasterrcnn_resnet50_fpn_v2", "outputs/pilot/faster_rcnn"),
        ("configs/pilot_retinanet.yaml", "retinanet_resnet50_fpn_v2", "outputs/pilot/retinanet"),
    ],
)
def test_pilot_configs_are_isolated(config_path, model_name, output_directory):
    config = load_config(config_path)

    assert config["run"]["type"] == "pilot"
    assert config["training"]["epochs"] == 3
    assert config["model"]["name"] == model_name
    assert config["output"]["directory"] == output_directory


@pytest.mark.parametrize(
    ("config_path", "model_name", "output_directory"),
    [
        (
            "configs/vietnam_v6_wikimedia_faster_rcnn.yaml",
            "fasterrcnn_resnet50_fpn_v2",
            "outputs/vietnam_pilot_v6_wikimedia/faster_rcnn/stage1",
        ),
        (
            "configs/vietnam_v6_wikimedia_retinanet.yaml",
            "retinanet_resnet50_fpn_v2",
            "outputs/vietnam_pilot_v6_wikimedia/retinanet/stage1",
        ),
    ],
)
def test_vietnam_v6_configs_are_self_contained_and_isolated(config_path, model_name, output_directory):
    config = load_config(config_path)

    assert config["model"]["name"] == model_name
    assert config["data"]["image_root"] == "data"
    assert config["sampling"]["enabled"] is False
    assert config["augmentation"]["color_jitter_probability"] == 0.0
    assert config["output"]["directory"] == output_directory
    assert config["training"]["epochs"] == 1
    assert config["training"]["freeze_backbone_body"] is True
    assert config["training"]["early_stop"]["patience"] == 2
