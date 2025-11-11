import subprocess
import shlex
from typing import Optional, Dict, Any
import requests
from app.config import JobProcessingError, CALLBACK_LOCALHOST_HOST
from urllib.parse import urlparse, urlunparse

def run(cmd: str, cwd: Optional[str]=None) -> None:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stdout}")

def ffprobe_duration(path: str) -> float:
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    try:
        return float(proc.stdout.strip())
    except:
        return 0.0


def ffprobe_duration(path: str) -> float:
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    try:
        return float(proc.stdout.strip())
    except:
        return 0.0

def mask_keep_intervals(in_wav: str, keep: list[tuple[float, float]], out_wav: str, sr: int = 48000, ac: int = 2):
    """
    keep 구간만 원본 레벨 그대로 두고, 그 외는 볼륨 0으로 '침묵화'하여 길이를 보존.
    ffmpeg volume 필터의 enable를 이용해 not(keep)을 0으로 만듦.
    """
    dur = ffprobe_duration(in_wav)
    if dur <= 0.0:
        # 입력이 이상하면 동일 길이 무음
        make_silence(out_wav, 0.0, ar=sr)
        return

    if not keep:
        # 전부 비-스피치: 전체 길이 무음
        run(f"ffmpeg -y -f lavfi -i anullsrc=channel_layout={'stereo' if ac==2 else 'mono'}:sample_rate={sr} -t {dur:.6f} -ar {sr} -ac {ac} {shlex.quote(out_wav)}")
        return

    expr = "+".join([f"between(t,{s:.6f},{e:.6f})" for s, e in keep])
    # keep 외 구간은 볼륨 0 → 타임라인 보존
    run(
        f'ffmpeg -y -i {shlex.quote(in_wav)} '
        f'-af "volume=0:enable=\'not({expr})\'" -ar {sr} -ac {ac} {shlex.quote(out_wav)}'
    )

def make_silence(path: str, seconds: float, ar: int = 24000):
    if seconds <= 0.0001:
        # 0초에 가까우면 빈 파일을 만들어 concat 오류 방지
        open(path, "wb").close()
        return
    run(f"ffmpeg -y -f lavfi -i anullsrc=r={ar}:cl=mono -t {seconds:.6f} -ar {ar} -ac 1 {shlex.quote(path)}")



def post_status(
    self,
    callback_url: str,
    status: str,
    *,
    result_key: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    stage_id: Optional[str] = None,
    stage_status: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    stage_id = stage_id or "pipeline"
    stage_status = stage_status or ("done" if status == "done" else "processing")

    payload: Dict[str, Any] = {
        "status": status,
        "stage_id": stage_id,
        "stage_status": stage_status,
    }
    if project_id is not None:
        payload["project_id"] = project_id
    if result_key is not None:
        payload["result_key"] = result_key
    if error is not None:
        payload["error"] = error
    if metadata is not None:
        payload["metadata"] = metadata

    target_url = __normalize_callback_url(callback_url)

    try:
        resp = self.http.post(target_url, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise JobProcessingError(f"Callback request failed: {exc}") from exc

    if not resp.ok:
        raise JobProcessingError(
            f"Callback responded with {resp.status_code}: {resp.text[:200]}"
        )


def __normalize_callback_url(call_back_url) -> str:
    parsed = urlparse(call_back_url)
    if (
        parsed.hostname in {"localhost", "127.0.0.1"}
        and CALLBACK_LOCALHOST_HOST
    ):
        host = CALLBACK_LOCALHOST_HOST
        netloc = host
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)