"""Temporal VLM Runtime의 analyze/evaluate 명령행 인터페이스입니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import IntervalRecord, evaluate_intervals
from .qwen_runtime import QwenTemporalRuntime


def analyze_video(args: argparse.Namespace) -> None:
    """영상 하나를 Qwen3-VL timeline JSON으로 변환합니다."""

    # 모델은 command 실행당 한 번만 로드해 video preprocessing과 generation에 재사용합니다.
    runtime = QwenTemporalRuntime(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    # 사용자 query와 sampling budget을 runtime에 전달해 timeline을 생성합니다.
    timeline = runtime.analyze(
        video=args.video,
        query=args.query,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    # 구조화 결과를 지정한 JSON 파일에 저장합니다.
    timeline.save(args.output)
    # 자동화에서도 확인하기 쉬운 완료 메시지를 출력합니다.
    print(f"wrote {len(timeline.events)} timeline events -> {args.output}")


def _load_intervals(path: Path) -> list[IntervalRecord]:
    # UTF-8 JSON 배열을 읽습니다.
    payload = json.loads(path.read_text(encoding="utf-8"))
    # evaluator 입력은 반드시 JSON array여야 합니다.
    if not isinstance(payload, list):
        raise ValueError(f"interval file must be a JSON array: {path}")
    # 각 object를 동일한 IntervalRecord schema로 변환합니다.
    return [IntervalRecord(**record) for record in payload]


def evaluate_files(args: argparse.Namespace) -> None:
    """ground truth와 prediction 시간 구간 파일을 평가합니다."""

    # ground-truth interval 목록을 읽습니다.
    ground_truth = _load_intervals(args.ground_truth)
    # prediction interval 목록을 읽습니다.
    predictions = _load_intervals(args.predictions)
    # lmms-eval temporal IoU 기반 metric을 계산합니다.
    metrics = evaluate_intervals(ground_truth, predictions)
    # shell에서 후처리하기 쉽도록 결과를 JSON으로 출력합니다.
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def create_parser() -> argparse.ArgumentParser:
    """analyze/evaluate subcommand parser를 생성합니다."""

    # 최상위 CLI parser를 생성합니다.
    parser = argparse.ArgumentParser(prog="temporal-vlm")
    # video inference와 metric evaluation을 별도 subcommand로 나눕니다.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 실제 Qwen3-VL video inference 명령을 정의합니다.
    analyze_parser = subparsers.add_parser("analyze", help="extract a structured timeline from video")
    # 분석할 로컬 video 파일을 필수 위치 인자로 받습니다.
    analyze_parser.add_argument("video", type=Path)
    # 어떤 사건/질문에 초점을 맞출지 자연어 query를 필수로 받습니다.
    analyze_parser.add_argument("--query", required=True)
    # 구조화 timeline JSON 저장 경로를 필수 옵션으로 받습니다.
    analyze_parser.add_argument("--output", type=Path, required=True)
    # 기존 VLM Reasoning Lab과도 이어지기 쉬운 Qwen3-VL 2B 모델을 기본값으로 둡니다.
    analyze_parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    # Qwen utility가 사용할 frame sampling FPS를 조절할 수 있게 합니다.
    analyze_parser.add_argument("--fps", type=float, default=2.0)
    # 긴 영상에서 무제한 frame sampling을 막기 위해 최대 frame 수를 노출합니다.
    analyze_parser.add_argument("--max-frames", type=int, default=256)
    # timeline JSON generation 길이를 필요에 따라 조절할 수 있습니다.
    analyze_parser.add_argument("--max-new-tokens", type=int, default=1024)
    # 실제 실행 함수를 subcommand에 연결합니다.
    analyze_parser.set_defaults(handler=analyze_video)

    # temporal interval benchmark 명령을 정의합니다.
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate temporal grounding intervals")
    # 정답 interval JSON 파일을 필수 옵션으로 받습니다.
    evaluate_parser.add_argument("--ground-truth", type=Path, required=True)
    # 모델 prediction interval JSON 파일을 필수 옵션으로 받습니다.
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    # 실제 실행 함수를 subcommand에 연결합니다.
    evaluate_parser.set_defaults(handler=evaluate_files)

    # 완성된 parser를 반환합니다.
    return parser


def main() -> None:
    """CLI entry point입니다."""

    # 현재 command line argument를 parser로 해석합니다.
    args = create_parser().parse_args()
    # 선택한 subcommand handler를 호출합니다.
    args.handler(args)


if __name__ == "__main__":
    # module 직접 실행도 동일하게 지원합니다.
    main()

