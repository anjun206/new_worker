# stt.py
import os
import json
import logging
import shutil
import subprocess
from pathlib import Path
import torch

# Transformers>=4.41 expects torch.utils._pytree.register_pytree_node.
_pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
if _pytree and not hasattr(_pytree, "register_pytree_node"):
    register_impl = getattr(_pytree, "_register_pytree_node", None)
    if register_impl:

        def register_pytree_node(
            node_type,
            flatten_fn,
            unflatten_fn,
            *,
            serialized_type_name=None,
            serialized_fields=None,
        ):
            """Transformers passes extra kwargs that the old torch implementation ignores."""
            return register_impl(node_type, flatten_fn, unflatten_fn)

        _pytree.register_pytree_node = register_pytree_node

import whisperx

try:
    from whisperx.diarize import DiarizationPipeline
except ImportError:  # WhisperX<3.7 fallback
    from whisperx import DiarizationPipeline
from config import WHISPERX_CACHE_DIR, ensure_job_dirs

logger = logging.getLogger(__name__)


def _whisperx_download_root(subdir: str) -> str:
    base = Path(WHISPERX_CACHE_DIR)
    path = base / subdir
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_asr(job_id: str):
    """입력 영상을 WhisperX로 전사하고 화자 분리를 수행합니다."""
    paths = ensure_job_dirs(job_id)
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 파일 경로 구성
    input_video = paths.input_dir / "source.mp4"
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found for job {job_id}")
    audio_path = paths.vid_speaks_dir / "audio.wav"

    # 1. 영상에서 오디오 추출 (Whisper 권장 형식: 모노 16kHz)
    # subprocess로 ffmpeg 실행 (사전 설치 필요)
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    subprocess.run(extract_cmd, check=True)

    # 2. WhisperX 모델을 불러와 전사 수행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading WhisperX ASR model (device=%s)", device)
    model = whisperx.load_model(
        "large-v2",
        device=device,
        download_root=_whisperx_download_root("asr"),
    )  # 정확도를 위해 large 모델 사용

    # 3. 단어 정렬 전 단계: 오디오 전사 후 구간 정보 확보
    audio = whisperx.load_audio(str(audio_path))
    logger.info("Running ASR transcription via WhisperX")
    result = model.transcribe(audio)
    segments = result["segments"]  # 텍스트와 대략적인 타임스탬프 포함

    # 4. 정밀한 타이밍을 위한 정렬 모델 로드
    logger.info("Loading alignment model for language=%s", result["language"])
    align_kwargs = {
        "language_code": result["language"],
        "device": device,
    }
    align_root = _whisperx_download_root("align")
    try:
        align_model, metadata = whisperx.load_align_model(
            download_root=align_root, **align_kwargs
        )
    except TypeError as exc:
        if "unexpected keyword argument 'download_root'" in str(exc):
            align_model, metadata = whisperx.load_align_model(**align_kwargs)
        else:
            raise
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
        logger.info("Initializing WhisperX diarization pipeline via pyannote")
        diarization_pipeline = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        diarization_segments = diarization_pipeline(str(audio_path))
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
    transcript_path = paths.src_sentence_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_segments, f, ensure_ascii=False, indent=2)
    shutil.copyfile(
        transcript_path, paths.outputs_text_dir / "src_transcript.json"
    )
    words_path = paths.src_words_dir / "aligned_segments.json"
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    return output_segments
