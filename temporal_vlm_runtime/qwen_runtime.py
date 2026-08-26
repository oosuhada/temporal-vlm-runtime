"""Qwen3-VL 공식 video processor 흐름을 구조화된 timeline 생성에 연결합니다."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info

from .timeline import TimelineDocument, parse_timeline_response


TIMELINE_SYSTEM_PROMPT = """
Analyze the video with precise temporal grounding.
Return JSON only. Do not use Markdown.
The JSON schema must be:
{
  "summary": "short overall answer",
  "events": [
    {
      "start_sec": 0.0,
      "end_sec": 1.0,
      "label": "short event label",
      "evidence": "visible evidence supporting the timestamp"
    }
  ]
}
Only include events that are supported by visible evidence in the video.
Use seconds as floating point numbers and keep events in chronological order.
""".strip()


class QwenTemporalRuntime:
    """Qwen3-VL을 한 번 로드한 뒤 여러 video query에 재사용하는 runtime입니다."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        max_new_tokens: int = 1024,
    ) -> None:
        # 실제 모델 checkpoint 이름을 결과 metadata와 재현성에 사용할 수 있도록 보관합니다.
        self.model_name = model_name
        # timeline JSON이 충분히 생성되면서도 과도한 출력 비용을 막도록 최대 생성 token 수를 저장합니다.
        self.max_new_tokens = max_new_tokens
        # Qwen3-VL 공식 Hugging Face processor를 그대로 로드합니다.
        self.processor = AutoProcessor.from_pretrained(model_name)
        # model architecture와 weight loading은 Transformers의 Qwen3-VL 구현을 그대로 사용합니다.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        # 추론 runtime이므로 dropout을 끄기 위해 eval 모드로 전환합니다.
        self.model.eval()

    def analyze(
        self,
        video: Path,
        query: str,
        fps: float = 2.0,
        max_frames: int = 256,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
    ) -> TimelineDocument:
        """한 영상과 자연어 query를 구조화된 TimelineDocument로 변환합니다."""

        # provenance를 유지하기 위해 로컬 video 경로를 절대 경로로 고정합니다.
        video_path = video.resolve()
        # 존재하지 않는 파일로 무거운 모델 추론을 시작하지 않도록 먼저 검사합니다.
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # Qwen3-VL이 처리할 video와 sampling budget을 공식 message 형식으로 구성합니다.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "fps": fps,
                        "max_frames": max_frames,
                        "total_pixels": total_pixels,
                        "min_pixels": min_pixels,
                    },
                    {
                        "type": "text",
                        "text": f"{TIMELINE_SYSTEM_PROMPT}\n\nUser query: {query}",
                    },
                ],
            }
        ]

        # chat template 문자열은 Qwen3-VL processor의 공식 formatting을 그대로 사용합니다.
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # video decoding, frame sampling, resize, metadata 생성은 bundled qwen-vl-utils에 그대로 위임합니다.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        # Qwen3-VL processor는 video tensor와 metadata를 분리해서 받으므로 utility 결과를 공식 cookbook 방식으로 풀어냅니다.
        if video_inputs is not None:
            # 각 video entry의 tensor와 metadata tuple을 분리합니다.
            video_tensors, video_metadatas = zip(*video_inputs)
            # processor가 기대하는 list 형태로 변환합니다.
            video_inputs = list(video_tensors)
            # timestamp alignment용 metadata도 동일하게 list로 변환합니다.
            video_metadatas = list(video_metadatas)
        else:
            # video 입력이 생성되지 않은 예외 상황에서는 metadata도 None으로 둡니다.
            video_metadatas = None

        # qwen-vl-utils가 이미 resize를 완료했으므로 do_resize=False로 중복 전처리를 막습니다.
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        )
        # Accelerate가 배치한 model device와 같은 장치로 processor Tensor를 이동합니다.
        inputs = inputs.to(self.model.device)

        # timeline 생성은 gradient가 필요하지 않으므로 inference mode로 실행합니다.
        with torch.inference_mode():
            # 실제 autoregressive generation은 Qwen3-VL model 구현을 그대로 사용합니다.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # prompt token을 제외하고 새로 생성된 assistant token만 잘라냅니다.
        generated_ids = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        # token ID를 사람이 읽을 수 있는 JSON 문자열로 decode합니다.
        response_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        # 자유 형식 model output을 엄격한 TimelineDocument schema로 검증해서 반환합니다.
        return parse_timeline_response(
            video=str(video_path),
            query=query,
            response_text=response_text,
        )

