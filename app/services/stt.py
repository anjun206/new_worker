# stt.py
import os
import json
import subprocess
import torch

# Transformers>=4.41 expects torch.utils._pytree.register_pytree_node.
_pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
if _pytree and not hasattr(_pytree, "register_pytree_node"):
    register_impl = getattr(_pytree, "_register_pytree_node", None)
    if register_impl:
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, *, serialized_type_name=None, serialized_fields=None):
            """Transformers passes extra kwargs that the old torch implementation ignores."""
            return register_impl(node_type, flatten_fn, unflatten_fn)

        _pytree.register_pytree_node = register_pytree_node

import whisperx

from whisperx import DiarizationPipeline


def run_asr(media_id: str, job_id: str, hf_token: str = None):
    """입력 영상을 WhisperX로 전사하고 화자 분리를 수행합니다."""
    # 파일 경로 구성
    input_video = os.path.join("inputs", media_id, "source.mp4")
    job_dir = os.path.join("interim", job_id)
    os.makedirs(job_dir, exist_ok=True)
    audio_path = os.path.join(job_dir, "audio.wav")

    # 1. 영상에서 오디오 추출 (Whisper 권장 형식: 모노 16kHz)
    # subprocess로 ffmpeg 실행 (사전 설치 필요)
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path,
    ]
    subprocess.run(extract_cmd, check=True)

    # 2. WhisperX 모델을 불러와 전사 수행
    device = "cuda" if whisperx.utils.get_device() == "cuda" else "cpu"
    model = whisperx.load_model(
        "large-v2", device=device
    )  # 정확도를 위해 large 모델 사용

    # 3. 단어 정렬 전 단계: 오디오 전사 후 구간 정보 확보
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)
    segments = result["segments"]  # 텍스트와 대략적인 타임스탬프 포함

    # 4. 정밀한 타이밍을 위한 정렬 모델 로드
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result_aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )
    segments = result_aligned["segments"]  # 단어 단위 타임스탬프가 포함된 구간

    # 5. pyannote 기반 화자 분리 (모델 접근을 위해 HF 토큰 필요)
    if hf_token:
        diarization_pipeline = DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarization_segments = diarization_pipeline(audio_path)
        # 각 구간에 화자 레이블 부여
        result_segments = whisperx.assign_word_speakers(
            diarization_segments, result_aligned
        )
        segments = result_segments["segments"]
        # 각 구간/단어에 speaker 키가 포함됨

    # 6. 응답용 구간 구조 단순화 (화자 단위로 텍스트 병합)
    output_segments = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        speaker = seg.get("speaker", "unknown_speaker")
        output_segments.append(
            {
                "speaker": (
                    f"SPEAKER_{speaker}" if isinstance(speaker, int) else str(speaker)
                ),
                "start": start,
                "end": end,
                "text": text,
            }
        )

    # 7. 전사 결과를 JSON으로 저장
    transcript_path = os.path.join(job_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_segments, f, ensure_ascii=False, indent=2)

    return output_segments
