from temporal_vlm_runtime.evaluation import IntervalRecord, evaluate_intervals
from temporal_vlm_runtime.timeline import parse_timeline_response


def test_parse_timeline_response_sorts_events() -> None:
    # 모델이 Markdown fence와 역순 event를 반환하는 흔한 형식을 준비합니다.
    response = """```json
    {
      "summary": "two events",
      "events": [
        {"start_sec": 8.0, "end_sec": 10.0, "label": "second", "evidence": "later"},
        {"start_sec": 1.0, "end_sec": 3.0, "label": "first", "evidence": "earlier"}
      ]
    }
    ```"""
    # parser가 실제 runtime metadata와 model JSON을 결합해 document를 생성합니다.
    document = parse_timeline_response("sample.mp4", "what happens?", response)
    # event가 timestamp 기준으로 자동 정렬됐는지 확인합니다.
    assert [event.label for event in document.events] == ["first", "second"]


def test_temporal_iou_metrics_use_prediction_ids() -> None:
    # 두 ground-truth 구간을 준비합니다.
    ground_truth = [
        IntervalRecord(id="a", start_sec=0.0, end_sec=10.0),
        IntervalRecord(id="b", start_sec=20.0, end_sec=30.0),
    ]
    # 순서를 뒤집되 동일 ID와 완전히 일치하는 prediction을 준비합니다.
    predictions = [
        IntervalRecord(id="b", start_sec=20.0, end_sec=30.0),
        IntervalRecord(id="a", start_sec=0.0, end_sec=10.0),
    ]
    # ID 기준 matching으로 metric을 계산합니다.
    metrics = evaluate_intervals(ground_truth, predictions)
    # 완전히 일치하는 구간이므로 평균 IoU는 1이어야 합니다.
    assert metrics["mean_iou"] == 1.0
    # 가장 엄격한 0.7 threshold recall도 100%여야 합니다.
    assert metrics["recall_iou_0.7"] == 1.0

