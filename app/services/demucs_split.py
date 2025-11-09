# demucs_split.py
import os
import subprocess
import shutil


def split_vocals(job_id: str):
    """Demucs 두 스템 모드로 보컬과 배경음을 분리합니다."""
    job_dir = os.path.join("interim", job_id)
    audio_path = os.path.join(job_dir, "audio.wav")
    output_dir = os.path.join(job_dir, "demucs_out")
    os.makedirs(output_dir, exist_ok=True)

    # CUDA를 사용해 Demucs 두 스템(보컬/배경) 분리 실행
    cmd = [
        "python3",
        "-m",
        "demucs.separate",
        "-d",
        "cuda",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "-o",
        output_dir,
        audio_path,
    ]
    subprocess.run(cmd, check=True)

    # Demucs는 모델 이름 폴더 하위에 결과를 생성함
    # 예: interim/<job_id>/demucs_out/htdemucs/audio/vocals.wav
    # 생성된 폴더를 찾아 실제 파일 경로 확정
    demucs_model_dir = os.path.join(output_dir, "htdemucs")
    # 입력 파일명(확장자 제거)과 동일한 폴더가 생성됨
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    sep_dir = os.path.join(demucs_model_dir, base_name)

    # 기대되는 출력 파일 경로 지정
    vocals_path = os.path.join(sep_dir, "vocals.wav")
    background_path = os.path.join(sep_dir, "no_vocals.wav")
    if not os.path.isfile(vocals_path) or not os.path.isfile(background_path):
        raise RuntimeError("Demucs output files not found")
    # interim 디렉터리로 복사해 이후 단계가 쉽게 접근하도록 함
    shutil.copy(vocals_path, os.path.join(job_dir, "vocals.wav"))
    shutil.copy(background_path, os.path.join(job_dir, "background.wav"))
    return {
        "vocals": os.path.join(job_dir, "vocals.wav"),
        "background": os.path.join(job_dir, "background.wav"),
    }
