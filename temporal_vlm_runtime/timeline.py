"""모델의 자유 형식 응답을 검증 가능한 timeline document로 변환합니다."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimelineEvent:
    """영상 내 하나의 시간 구간 사건을 표현합니다."""

    start_sec: float
    end_sec: float
    label: str
    evidence: str

    def __post_init__(self) -> None:
        # 음수 timestamp는 실제 영상 시간으로 사용할 수 없으므로 거부합니다.
        if self.start_sec < 0 or self.end_sec < 0:
            raise ValueError("event timestamps must be non-negative")
        # 사건 구간은 반드시 시작보다 종료 시간이 뒤에 있어야 합니다.
        if self.end_sec <= self.start_sec:
            raise ValueError("event end_sec must be greater than start_sec")
        # label이 비어 있으면 timeline을 후처리하기 어려우므로 필수로 검증합니다.
        if not self.label.strip():
            raise ValueError("event label must not be empty")


@dataclass(frozen=True)
class TimelineDocument:
    """한 영상과 query에 대한 전체 구조화 결과입니다."""

    video: str
    query: str
    summary: str
    events: list[TimelineEvent]

    def to_dict(self) -> dict[str, object]:
        # dataclass 내부 event까지 JSON 직렬화 가능한 기본 타입으로 변환합니다.
        return asdict(self)

    def save(self, path: Path) -> None:
        # 출력 폴더가 없으면 자동 생성합니다.
        path.parent.mkdir(parents=True, exist_ok=True)
        # 사람이 직접 확인하기 쉬운 pretty JSON으로 저장합니다.
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _strip_markdown_fence(text: str) -> str:
    # 모델이 JSON을 markdown code fence로 감싼 경우 fence만 제거합니다.
    stripped = text.strip()
    # 시작 fence가 json 또는 일반 code block인지 확인합니다.
    if stripped.startswith("```"):
        # 첫 줄의 ```json 또는 ```를 제거합니다.
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        # 마지막 closing fence도 제거합니다.
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    # 정리된 JSON 후보 문자열을 반환합니다.
    return stripped.strip()


def _extract_json_object(text: str) -> dict[str, object]:
    # 먼저 markdown fence만 제거한 전체 문자열을 JSON으로 읽어봅니다.
    candidate = _strip_markdown_fence(text)
    try:
        # 정상적인 JSON-only 응답이면 추가 보정 없이 바로 반환합니다.
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        # 모델이 앞뒤 설명을 붙인 경우 가장 바깥 JSON object 구간을 찾아 재시도합니다.
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        # JSON object 자체가 없으면 잘못된 구조화 응답으로 처리합니다.
        if match is None:
            raise ValueError("model response does not contain a JSON object")
        # 찾은 object substring만 JSON으로 파싱합니다.
        payload = json.loads(match.group(0))

    # timeline root는 object여야 하므로 list나 scalar 응답을 거부합니다.
    if not isinstance(payload, dict):
        raise ValueError("timeline response root must be a JSON object")
    # 타입 검사를 통과한 payload를 반환합니다.
    return payload


def parse_timeline_response(video: str, query: str, response_text: str) -> TimelineDocument:
    """Qwen3-VL 응답을 TimelineDocument로 검증/변환합니다."""

    # 자유 형식 model response에서 JSON object를 추출합니다.
    payload = _extract_json_object(response_text)
    # summary는 없더라도 빈 문자열로 안전하게 처리합니다.
    summary = str(payload.get("summary", "")).strip()
    # events는 반드시 배열이어야 구조화된 timeline으로 사용할 수 있습니다.
    raw_events = payload.get("events", [])
    # 잘못된 events 타입을 즉시 검출합니다.
    if not isinstance(raw_events, list):
        raise ValueError("timeline 'events' must be an array")

    # 검증된 TimelineEvent 목록을 준비합니다.
    events: list[TimelineEvent] = []
    # 모델이 반환한 각 event object를 엄격한 dataclass로 변환합니다.
    for raw_event in raw_events:
        # event 하나는 JSON object여야 합니다.
        if not isinstance(raw_event, dict):
            raise ValueError("each timeline event must be a JSON object")
        # 숫자 timestamp와 문자열 설명을 명시적으로 변환합니다.
        event = TimelineEvent(
            start_sec=float(raw_event["start_sec"]),
            end_sec=float(raw_event["end_sec"]),
            label=str(raw_event["label"]),
            evidence=str(raw_event.get("evidence", "")),
        )
        # timestamp 순서 기준으로 정렬하기 전에 event 목록에 추가합니다.
        events.append(event)

    # 모델이 event 순서를 섞어 반환해도 timeline은 시간순으로 고정합니다.
    events.sort(key=lambda event: (event.start_sec, event.end_sec))
    # 실제 입력 video/query를 model output이 아닌 runtime metadata에서 강제로 넣어 provenance를 유지합니다.
    return TimelineDocument(
        video=str(Path(video).resolve()),
        query=query,
        summary=summary,
        events=events,
    )

