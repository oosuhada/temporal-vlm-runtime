"""lmms-eval에서 가져온 temporal IoU 함수를 이용해 timeline 구간 정확도를 검증합니다."""

from __future__ import annotations

from dataclasses import dataclass

from third_party.lmms_eval.charades_sta.eval_tvg import iou


@dataclass(frozen=True)
class IntervalRecord:
    """평가용 ground-truth 또는 prediction 시간 구간입니다."""

    id: str
    start_sec: float
    end_sec: float

    def as_pair(self) -> list[float]:
        # lmms-eval의 iou 함수가 기대하는 [start, end] 형식으로 변환합니다.
        return [float(self.start_sec), float(self.end_sec)]


def evaluate_intervals(
    ground_truth: list[IntervalRecord],
    predictions: list[IntervalRecord],
) -> dict[str, float]:
    """ID가 같은 interval끼리 temporal IoU와 threshold recall을 계산합니다."""

    # prediction을 ID 기준 dict로 만들어 순서 차이에 영향을 받지 않게 합니다.
    predictions_by_id = {record.id: record for record in predictions}
    # 모든 ground-truth sample의 IoU를 저장할 리스트를 준비합니다.
    ious: list[float] = []

    # ground-truth ID를 기준으로 prediction을 찾아 IoU를 계산합니다.
    for truth in ground_truth:
        # 해당 ID prediction이 없으면 temporal grounding 실패로 보고 IoU 0을 기록합니다.
        prediction = predictions_by_id.get(truth.id)
        # 누락 prediction은 0점으로 처리하고 다음 sample로 넘어갑니다.
        if prediction is None:
            ious.append(0.0)
            continue
        # 실제 IoU 공식은 lmms-eval Charades-STA evaluator의 구현을 그대로 호출합니다.
        ious.append(float(iou(truth.as_pair(), prediction.as_pair())))

    # 비어 있는 benchmark를 0으로 나누지 않도록 명확히 거부합니다.
    if not ious:
        raise ValueError("ground_truth must contain at least one interval")

    # Charades-STA에서 흔히 사용하는 threshold별 recall을 계산합니다.
    recall_03 = sum(value >= 0.3 for value in ious) / len(ious)
    # 더 엄격한 0.5 threshold recall도 계산합니다.
    recall_05 = sum(value >= 0.5 for value in ious) / len(ious)
    # 높은 정밀 temporal grounding을 보는 0.7 threshold recall도 계산합니다.
    recall_07 = sum(value >= 0.7 for value in ious) / len(ious)
    # 전체 sample의 평균 IoU를 계산합니다.
    mean_iou = sum(ious) / len(ious)
    # CLI/JSON에서 그대로 사용할 수 있는 metric dict를 반환합니다.
    return {
        "samples": float(len(ious)),
        "mean_iou": mean_iou,
        "recall_iou_0.3": recall_03,
        "recall_iou_0.5": recall_05,
        "recall_iou_0.7": recall_07,
    }

