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
from services.demucs_split import split_vocals

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
    raw_audio_path = paths.vid_speaks_dir / "audio.wav"

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
        str(raw_audio_path),
    ]
    subprocess.run(extract_cmd, check=True)

    # 1-1. Demucs로 보컬/배경을 분리해 보컬만 ASR에 사용
    demucs_result = split_vocals(job_id)
    vocals_audio_path = Path(demucs_result["vocals"])

    # 2. WhisperX 모델을 불러와 전사 수행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading WhisperX ASR model (device=%s)", device)
    model = whisperx.load_model(
        "large-v2",
        device=device,
        download_root=_whisperx_download_root("asr"),
    )  # 정확도를 위해 large 모델 사용

    # 3. 단어 정렬 전 단계: 오디오 전사 후 구간 정보 확보
    audio = whisperx.load_audio(str(vocals_audio_path))
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
        diarization_segments = diarization_pipeline(str(vocals_audio_path))
        # 각 구간에 화자 레이블 부여
        result_segments = whisperx.assign_word_speakers(
            diarization_segments, result_aligned
        )
        segments = result_segments["segments"]
        # 각 구간/단어에 speaker 키가 포함됨

    # 6. 문장/단어 메타데이터 정리 및 저장
    word_dir = paths.src_words_dir
    for existing in word_dir.glob("segment_*_words.json"):
        existing.unlink()

    output_segments = []
    prev_end = None
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "").strip()
        raw_speaker = seg.get("speaker", "unknown_speaker")
        speaker = (
            f"SPEAKER_{raw_speaker}"
            if isinstance(raw_speaker, int)
            else str(raw_speaker)
        )
        duration = round(max(0.0, end - start), 3)
        next_start = (
            float(segments[idx + 1].get("start"))
            if idx + 1 < len(segments) and segments[idx + 1].get("start") is not None
            else None
        )
        gap_after = round(next_start - end, 3) if next_start is not None else None
        gap_after_vad = round(max(0.0, gap_after), 3) if gap_after is not None else None
        is_overlapping = prev_end is not None and start < prev_end
        prev_end = end if prev_end is None else max(prev_end, end)

        segment_uid = f"segment_{idx:04d}"
        words = seg.get("words") or []
        word_ids = []
        word_entries = []
        for w_idx, word in enumerate(words):
            word_start = word.get("start")
            word_end = word.get("end")
            word_duration = (
                round(max(0.0, word_end - word_start), 3)
                if word_start is not None and word_end is not None
                else None
            )
            word_id = f"{segment_uid}_word_{w_idx:03d}"
            raw_word_speaker = word.get("speaker")
            if isinstance(raw_word_speaker, int):
                word_speaker = f"SPEAKER_{raw_word_speaker}"
            elif raw_word_speaker:
                word_speaker = str(raw_word_speaker)
            else:
                word_speaker = speaker
            word_ids.append(word_id)
            word_entries.append(
                {
                    "segment_id": word_id,
                    "sentence_segment_id": segment_uid,
                    "start": word_start,
                    "end": word_end,
                    "text": (word.get("word") or "").strip(),
                    "duration": word_duration,
                    "speaker": word_speaker,
                }
            )

        word_file = word_dir / f"{segment_uid}_words.json"
        with open(word_file, "w", encoding="utf-8") as wf:
            json.dump(word_entries, wf, ensure_ascii=False, indent=2)

        output_segments.append(
            {
                "segment_id": segment_uid,
                "original_segment_id": seg.get("id", idx),
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "duration": duration,
                "gap_after": gap_after,
                "gap_after_vad": gap_after_vad,
                "word_ids": word_ids,
                "word_metadata_file": word_file.name,
                "is_overlapping": is_overlapping,
                "asr_score": seg.get("avg_logprob"),
            }
        )

    # 7. 전사 결과를 JSON으로 저장
    transcript_path = paths.src_sentence_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_segments, f, ensure_ascii=False, indent=2)
    shutil.copyfile(transcript_path, paths.outputs_text_dir / "src_transcript.json")
    words_path = paths.src_words_dir / "aligned_segments.json"
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    return output_segments
