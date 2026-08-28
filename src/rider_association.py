"""Liên kết box đầu với vùng xe mà không suy đoán quá mức về vai trò.

Module này là baseline cho demo. Detector hiện tại chỉ có ba lớp độc lập,
không có ID người/xe hoặc nhãn tài xế. Vì vậy ``driver_candidate`` chỉ có
nghĩa là một đầu duy nhất được ghép rõ ràng với một vùng xe; nó không phải
khẳng định chắc chắn người đó là tài xế và không được dùng để phát cảnh báo
vi phạm chính thức.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils import load_yaml, resolve_project_path


VEHICLE_CLASS = "BikeWithRider"
HEAD_CLASSES = {"Helmet": "helmet", "NoHelmet": "no_helmet"}


@dataclass(frozen=True)
class AssociationConfig:
    """Ngưỡng hình học của baseline; sẽ chỉ được điều chỉnh bằng role_dev."""

    min_head_coverage: float = 0.60
    ambiguity_margin: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_head_coverage <= 1.0:
            raise ValueError("min_head_coverage phải nằm trong [0, 1]")
        if self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin không được âm")


DEFAULT_ASSOCIATION_CONFIG = AssociationConfig()


@dataclass(frozen=True)
class RoleDecisionConfig:
    """Điều kiện bật quy tắc vai trò đã được duyệt trên role_dev."""

    enabled: bool = False
    single_head_rule: bool = False
    multihead_strategy: str = "abstain"
    minimum_precision: float = 0.95
    minimum_support: int = 50
    observed_precision: float | None = None
    observed_recall: float | None = None
    observed_support: int = 0
    source_split: str = "validation"
    source_tasks: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_precision", self.minimum_precision),
            ("observed_precision", self.observed_precision),
            ("observed_recall", self.observed_recall),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong [0, 1]")
        if self.minimum_support < 1 or self.observed_support < 0:
            raise ValueError("support của quy tắc vai trò không hợp lệ")
        if self.multihead_strategy != "abstain":
            raise ValueError("Phiên bản hiện tại chỉ hỗ trợ multihead_strategy=abstain")

    @property
    def single_head_ready(self) -> bool:
        return bool(
            self.enabled
            and self.single_head_rule
            and self.source_split == "validation"
            and self.observed_precision is not None
            and self.observed_precision >= self.minimum_precision
            and self.observed_support >= self.minimum_support
        )


DEFAULT_ROLE_DECISION_CONFIG = RoleDecisionConfig()


def load_role_decision_config(path: str | Path) -> tuple[AssociationConfig, RoleDecisionConfig]:
    """Đọc cấu hình association/role và kiểm tra điều kiện an toàn."""
    payload = load_yaml(path)
    association = payload.get("association", {})
    role = payload.get("role_decision", {})
    if not isinstance(association, Mapping) or not isinstance(role, Mapping):
        raise ValueError("Cấu hình rider_association cần hai mapping association và role_decision")
    association_config = AssociationConfig(
            min_head_coverage=float(association.get("min_head_coverage", 0.60)),
            ambiguity_margin=float(association.get("ambiguity_margin", 0.08)),
        )
    role_config = RoleDecisionConfig(
            enabled=bool(role.get("enabled", False)),
            single_head_rule=bool(role.get("single_head_rule", False)),
            multihead_strategy=str(role.get("multihead_strategy", "abstain")),
            minimum_precision=float(role.get("minimum_precision", 0.95)),
            minimum_support=int(role.get("minimum_support", 50)),
            observed_precision=(float(role["observed_precision"]) if role.get("observed_precision") is not None else None),
            observed_recall=(float(role["observed_recall"]) if role.get("observed_recall") is not None else None),
            observed_support=int(role.get("observed_support", 0)),
            source_split=str(role.get("source_split", "validation")),
            source_tasks=(str(role["source_tasks"]) if role.get("source_tasks") else None),
            source_sha256=(str(role["source_sha256"]) if role.get("source_sha256") else None),
        )
    if role_config.enabled:
        if not role_config.source_tasks or not role_config.source_sha256:
            raise ValueError("Quy tắc role bật nhưng thiếu source_tasks/source_sha256")
        source_path = resolve_project_path(role_config.source_tasks)
        if not source_path.is_file():
            raise ValueError(f"Không tìm thấy role_dev đã đóng băng: {source_path}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash.casefold() != role_config.source_sha256.casefold():
            raise ValueError("SHA-256 role_dev không khớp cấu hình rider_association")
    return association_config, role_config


def _box(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = record.get("box")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("Mỗi detection phải có box [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection có box rỗng hoặc đảo tọa độ")
    return x1, y1, x2, y2


def _detection_id(record: Mapping[str, Any], index: int) -> str:
    value = record.get("detection_id", f"detection_{index + 1}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("detection_id phải là chuỗi không rỗng")
    return value


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _inside_area_ratio(
    head_box: tuple[float, float, float, float],
    vehicle_box: tuple[float, float, float, float],
) -> float:
    hx1, hy1, hx2, hy2 = head_box
    vx1, vy1, vx2, vy2 = vehicle_box
    overlap_width = max(0.0, min(hx2, vx2) - max(hx1, vx1))
    overlap_height = max(0.0, min(hy2, vy2) - max(hy1, vy1))
    overlap = overlap_width * overlap_height
    head_area = (hx2 - hx1) * (hy2 - hy1)
    return overlap / head_area


def _center_is_inside(
    head_box: tuple[float, float, float, float],
    vehicle_box: tuple[float, float, float, float],
) -> bool:
    x, y = _center(head_box)
    x1, y1, x2, y2 = vehicle_box
    return x1 <= x <= x2 and y1 <= y <= y2


def _proximity_to_vehicle_center(
    head_box: tuple[float, float, float, float],
    vehicle_box: tuple[float, float, float, float],
) -> float:
    """Điểm phụ chỉ để phá thế hòa; không suy ra đầu xe hay vai trò."""
    head_x, head_y = _center(head_box)
    vehicle_x, vehicle_y = _center(vehicle_box)
    width = vehicle_box[2] - vehicle_box[0]
    height = vehicle_box[3] - vehicle_box[1]
    normalized_distance = ((head_x - vehicle_x) / width) ** 2 + ((head_y - vehicle_y) / height) ** 2
    return max(0.0, 1.0 - normalized_distance**0.5)


def association_score(
    head_box: tuple[float, float, float, float],
    vehicle_box: tuple[float, float, float, float],
    config: AssociationConfig = DEFAULT_ASSOCIATION_CONFIG,
) -> float | None:
    """Trả điểm ghép đầu–xe hoặc ``None`` khi không đạt quy tắc hình học."""
    coverage = _inside_area_ratio(head_box, vehicle_box)
    if not _center_is_inside(head_box, vehicle_box) or coverage < config.min_head_coverage:
        return None
    return 0.90 * coverage + 0.10 * _proximity_to_vehicle_center(head_box, vehicle_box)


def analyze_rider_roles(
    detections: Sequence[Mapping[str, Any]],
    config: AssociationConfig = DEFAULT_ASSOCIATION_CONFIG,
    role_config: RoleDecisionConfig = DEFAULT_ROLE_DECISION_CONFIG,
) -> dict[str, Any]:
    """Ghép đầu–xe theo hình học và trả schema an toàn cho API/frontend.

    Mọi vai trò ở baseline này là ``driver_candidate`` hoặc ``unknown``. Khi
    vùng xe có từ hai đầu trở lên, hàm cố ý không chọn tài xế dựa trên vị trí
    trái/phải vì detector không cung cấp hướng di chuyển hoặc đầu xe.
    """
    normalized: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for index, raw_record in enumerate(detections):
        if not isinstance(raw_record, Mapping):
            raise TypeError("detections phải là danh sách mapping")
        detection_id = _detection_id(raw_record, index)
        if detection_id in known_ids:
            raise ValueError(f"detection_id bị trùng: {detection_id}")
        known_ids.add(detection_id)
        class_name = raw_record.get("class_name")
        if not isinstance(class_name, str):
            raise ValueError("Mỗi detection phải có class_name là chuỗi")
        normalized.append(
            {
                "detection_id": detection_id,
                "class_name": class_name,
                "box": list(_box(raw_record)),
                "confidence": raw_record.get("confidence"),
            }
        )

    vehicles = [record for record in normalized if record["class_name"] == VEHICLE_CLASS]
    heads = [record for record in normalized if record["class_name"] in HEAD_CLASSES]
    group_heads: dict[str, list[dict[str, Any]]] = {record["detection_id"]: [] for record in vehicles}
    unassigned_heads: list[dict[str, Any]] = []
    ambiguous_heads: list[dict[str, Any]] = []

    for head in heads:
        head_box = tuple(head["box"])
        ranked: list[tuple[float, dict[str, Any]]] = []
        for vehicle in vehicles:
            score = association_score(head_box, tuple(vehicle["box"]), config)
            if score is not None:
                ranked.append((score, vehicle))
        ranked.sort(key=lambda item: (-item[0], item[1]["detection_id"]))

        head_payload = {
            "head_detection_id": head["detection_id"],
            "helmet_status": HEAD_CLASSES[head["class_name"]],
            "head_box": head["box"],
        }
        if not ranked:
            unassigned_heads.append({**head_payload, "reason": "no_vehicle_match"})
            continue
        best_score, best_vehicle = ranked[0]
        if len(ranked) > 1 and best_score - ranked[1][0] <= config.ambiguity_margin:
            ambiguous_heads.append(
                {
                    **head_payload,
                    "reason": "multiple_vehicle_matches",
                    "candidate_vehicle_detection_ids": [vehicle["detection_id"] for _score, vehicle in ranked],
                }
            )
            continue
        group_heads[best_vehicle["detection_id"]].append({**head_payload, "association_score": round(best_score, 6)})

    rider_groups: list[dict[str, Any]] = []
    driver_candidate_no_helmet = 0
    rule_based_drivers = 0
    driver_no_helmet_alerts = 0
    for group_index, vehicle in enumerate(vehicles, start=1):
        attached_heads = sorted(group_heads[vehicle["detection_id"]], key=lambda item: item["head_detection_id"])
        driver: dict[str, Any] | None = None
        if len(attached_heads) == 1:
            candidate = attached_heads[0]
            if role_config.single_head_ready:
                driver = {
                    "head_detection_id": candidate["head_detection_id"],
                    "helmet_status": candidate["helmet_status"],
                    "role": "driver",
                    "status": "rule_based",
                    "reason": "validated_single_head_rule",
                    "validation_evidence": {
                        "split": role_config.source_split,
                        "precision": role_config.observed_precision,
                        "recall": role_config.observed_recall,
                        "support": role_config.observed_support,
                        "source_tasks": role_config.source_tasks,
                    },
                }
                rule_based_drivers += 1
                if candidate["helmet_status"] == "no_helmet":
                    driver_no_helmet_alerts += 1
            else:
                driver = {
                    "head_detection_id": candidate["head_detection_id"],
                    "helmet_status": candidate["helmet_status"],
                    "role": "driver_candidate",
                    "status": "candidate_only",
                    "reason": "single_head_associated_with_vehicle",
                }
            if candidate["helmet_status"] == "no_helmet":
                driver_candidate_no_helmet += 1
        rider_groups.append(
            {
                "group_id": f"rider_group_{group_index}",
                "bike_detection_id": vehicle["detection_id"],
                "bike_box": vehicle["box"],
                "association_status": "associated" if attached_heads else "no_associated_head",
                "heads": attached_heads,
                "driver": driver,
            }
        )

    return {
        "version": "rider_role_rule_v2" if role_config.single_head_ready else "association_baseline_v1",
        "role_inference_status": "rule_based_with_abstention" if role_config.single_head_ready else "candidate_only",
        "rider_groups": rider_groups,
        "unassigned_heads": unassigned_heads,
        "ambiguous_heads": ambiguous_heads,
        "summary": {
            "vehicles": len(vehicles),
            "heads": len(heads),
            "associated_heads": sum(len(group["heads"]) for group in rider_groups),
            "unassigned_heads": len(unassigned_heads),
            "ambiguous_heads": len(ambiguous_heads),
            "driver_candidates": sum(1 for group in rider_groups if group["driver"] is not None),
            "driver_candidate_no_helmet": driver_candidate_no_helmet,
            "rule_based_drivers": rule_based_drivers,
            "driver_no_helmet_alerts": driver_no_helmet_alerts,
            "unknown_role_groups": sum(1 for group in rider_groups if group["heads"] and group["driver"] is None),
            "confirmed_driver_no_helmet": 0,
        },
        "limitations": [
            "Baseline chỉ ghép đầu với vùng xe bằng hình học.",
            "driver_candidate không phải kết luận chắc chắn về tài xế và không tạo cảnh báo vi phạm chính thức.",
            "Nhóm nhiều đầu hoặc ghép với nhiều xe được giữ ở trạng thái mơ hồ cho tới khi có nhãn role_dev.",
            "Quy tắc một-đầu được chọn trên role_dev validation; số liệu quan sát không phải xác suất đúng của từng ảnh.",
            "Chưa có role_test độc lập để báo cáo chất lượng cuối cùng.",
        ],
    }
