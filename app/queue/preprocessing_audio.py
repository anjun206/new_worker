import os
import shlex
import os
from typing import Tuple, Dict, List
from app.config import run
import importlib.util
import glob
from app.config import ffprobe_duration
from app.services import compute_vad_silences


def _annotate_segments(segments: List[Dict]) -> List[Dict]:
    """
    Ensure downstream metadata consumers receive normalized segment fields.
    """
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        seg["seg_id"] = seg.get("seg_id", idx)
        seg["length"] = max(0.0, end - start)
        seg.setdefault("issues", [])
        seg.setdefault("score", None)
    return segments

def separate_bgm_vocals(in_wav: str, out_vocals: str, out_bgm: str, model: str = "htdemucs"):
    """
    Demucs 2-stems 분리: vocals / no_vocals
    demucs 미설치 또는 실행 실패 시 즉시 예외를 발생시킨다.
    """
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError("demucs package not installed; background separation is unavailable")

    outdir = os.path.join(os.path.dirname(out_vocals), "sep")
    try:
        run(
            f"python -m demucs.separate -n {model} --two-stems=vocals -o {shlex.quote(outdir)} {shlex.quote(in_wav)}"
        )
        base = os.path.splitext(os.path.basename(in_wav))[0]
        cand_dir = glob.glob(os.path.join(outdir, model, base))
        if not cand_dir:
            raise RuntimeError("demucs output not found")
        cand_dir = cand_dir[0]
        v = glob.glob(os.path.join(cand_dir, "vocals.wav"))[0]
        nv = glob.glob(os.path.join(cand_dir, "no_vocals.wav"))[0]
        run(f"ffmpeg -y -i {shlex.quote(v)} -ar 48000 -ac 2 {shlex.quote(out_vocals)}")
        run(f"ffmpeg -y -i {shlex.quote(nv)} -ar 48000 -ac 2 {shlex.quote(out_bgm)}")
    except Exception as exc:
        raise RuntimeError("demucs separation failed") from exc



def extract_audio_full(video_in: str, wav_out: str):
    # 원본 오디오를 48k 스테레오 wav로 추출
    run(f"ffmpeg -y -i {shlex.quote(video_in)} -map 0:a:0 -ac 2 -ar 48000 -vn -c:a pcm_s16le {shlex.quote(wav_out)}")

def extract_tracks(in_path: str, work: str) -> Tuple[str, str, str, str]:
    """
    전체 오디오(48k) 추출 → 보이스/배경 분리 → 보이스 16k/mono까지 반환
    returns: (full_48k, vocals_48k, bgm_48k, vocals_16k_raw)
    """
    full_48k = os.path.join(work, "audio_full_48k.wav")
    extract_audio_full(in_path, full_48k)

    vocals_48k = os.path.join(work, "vocals_48k.wav")
    bgm_48k = os.path.join(work, "bgm_48k.wav")
    if os.getenv("SEPARATE_BGM", "1") == "1":
        separate_bgm_vocals(full_48k, vocals_48k, bgm_48k)
    else:
        run(
            f"ffmpeg -y -i {shlex.quote(full_48k)} -ar 48000 -ac 2 {shlex.quote(vocals_48k)}"
        )
        run(
            f"ffmpeg -y -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 -t {ffprobe_duration(full_48k):.3f} {shlex.quote(bgm_48k)}"
        )

    vocals_16k_raw = os.path.join(work, "vocals_16k_raw.wav")
    run(
        f"ffmpeg -y -i {shlex.quote(vocals_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(vocals_16k_raw)}"
    )
    return full_48k, vocals_48k, bgm_48k, vocals_16k_raw


def ensure_workdir(job_id: str) -> str:
    workdir = os.path.join("/app/data", job_id)
    os.makedirs(workdir, exist_ok=True)
    return workdir


def run_prepare_stage(job_id: str, input_path: str) -> Dict[str, Any]:
    workdir = ensure_workdir(job_id)
    full_48k, vocals_48k, bgm_48k, vocals_16k_raw = _extract_tracks(input_path, workdir)
    total = ffprobe_duration(full_48k)

    silences = compute_vad_silences(
        vocals_16k_raw,
        aggressiveness=int(os.getenv("VAD_AGGR", "3")),
        frame_ms=int(os.getenv("VAD_FRAME_MS", "30")),
    )

    segments = _whisper_transcribe(vocals_16k_raw)

    margin = float(os.getenv("STT_INTERVAL_MARGIN", "0.10"))
    stt_intervals = merge_intervals(
        [
            (
                max(0.0, float(s["start"]) - margin),
                min(float(total), float(s["end"]) + margin),
            )
            for s in segments
            if float(s["end"]) > float(s["start"])
        ]
    )

    speech_only_48k = os.path.join(workdir, "speech_only_48k.wav")
    vocals_fx_48k = os.path.join(workdir, "vocals_fx_48k.wav")
    mask_keep_intervals(vocals_48k, stt_intervals, speech_only_48k, sr=48000, ac=2)
    nonspeech_intervals = complement_intervals(stt_intervals, total)
    mask_keep_intervals(vocals_48k, nonspeech_intervals, vocals_fx_48k, sr=48000, ac=2)

    wav_16k = os.path.join(workdir, "speech_16k.wav")
    run(
        f"ffmpeg -y -i {shlex.quote(speech_only_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(wav_16k)}"
    )

    for i in range(len(segments)):
        if i < len(segments) - 1:
            st = float(segments[i]["end"])
            en = float(segments[i + 1]["start"])
            segments[i]["gap_after_vad"] = sum_silence_between(silences, st, en)
            segments[i]["gap_after"] = max(0.0, en - st)
        else:
            segments[i]["gap_after_vad"] = 0.0
            segments[i]["gap_after"] = 0.0

    _annotate_segments(segments)

    meta = {
        "job_id": job_id,
        "workdir": workdir,
        "input": input_path,
        "audio_full_48k": full_48k,
        "vocals_48k": vocals_48k,
        "bgm_48k": bgm_48k,
        "speech_only_48k": speech_only_48k,
        "vocals_fx_48k": vocals_fx_48k,
        "wav_16k": wav_16k,
        "orig_duration": total,
        "segments": segments,
        "silences": silences,
        "speech_intervals_stt": stt_intervals,
        "nonspeech_intervals_stt": nonspeech_intervals,
    }
    save_meta(workdir, meta)
    return meta