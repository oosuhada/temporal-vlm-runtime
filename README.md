# Temporal VLM Runtime

A clean-history video understanding runtime built from Qwen3-VL's released video utility and fine-tuning source.

Instead of rebuilding a video VLM, this repository turns the existing Qwen3-VL video pipeline into one focused capability: **extracting a machine-readable timeline of events with timestamp evidence**.

```text
Video
  ↓
Qwen3-VL video processor / qwen-vl-utils
  ↓
Visual tokens + timestamp alignment
  ↓
Qwen3-VL generation
  ↓
Validated timeline JSON
  ↓
Temporal IoU evaluation
```

## What is added here

- `temporal-vlm analyze`: produce structured event timelines from a video.
- Strict JSON parser and schema validation for model output.
- Query-focused temporal grounding prompts.
- `temporal-vlm evaluate`: evaluate predicted event intervals using the temporal IoU logic retained from `lmms-eval`.
- Qwen3-VL fine-tuning source remains included under `qwen-vl-finetune` for later domain adaptation without recreating a training stack.

## Install

The Qwen video utility source is included directly in this repository and installed locally.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ./qwen-vl-utils
pip install -e .
```

For faster video decoding, install the optional decoder recommended by Qwen:

```bash
pip install -e "./qwen-vl-utils[decord]"
```

## Analyze a video

```bash
temporal-vlm analyze ./sample.mp4 \
  --query "When does the person begin preparing the drink, and what happens afterward?" \
  --output ./artifacts/sample.timeline.json
```

Default model:

```text
Qwen/Qwen3-VL-2B-Instruct
```

You can replace it with another compatible Qwen3-VL checkpoint using `--model`.

## Output

```json
{
  "video": "/absolute/path/sample.mp4",
  "query": "When does the person begin preparing the drink, and what happens afterward?",
  "summary": "The person prepares the drink and then serves it.",
  "events": [
    {
      "start_sec": 4.2,
      "end_sec": 9.8,
      "label": "drink preparation",
      "evidence": "The person picks up ingredients and mixes them."
    }
  ]
}
```

## Evaluate temporal grounding

Ground truth and prediction files use the same simple interval schema:

```json
[
  {"id": "sample-001", "start_sec": 4.0, "end_sec": 10.0}
]
```

```bash
temporal-vlm evaluate \
  --ground-truth ./ground_truth.json \
  --predictions ./predictions.json
```

## Source lineage

See [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md). No upstream Git history is included.

