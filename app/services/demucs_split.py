# demucs_split.py
import os
import subprocess
import shutil


def split_vocals(job_id: str):
    """Separate vocals from background using Demucs (two-stems mode)."""
    job_dir = os.path.join("interim", job_id)
    audio_path = os.path.join(job_dir, "audio.wav")
    output_dir = os.path.join(job_dir, "demucs_out")
    os.makedirs(output_dir, exist_ok=True)

    # Run Demucs in two-stems (vocals vs other) mode. Using -d cuda for GPU.
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

    # Demucs will output files in a model-named subdirectory
    # e.g., interim/<job_id>/demucs_out/htdemucs/audio/vocals.wav and no_vocals.wav
    # Locate the output folder created by Demucs
    demucs_model_dir = os.path.join(output_dir, "htdemucs")
    # It should contain a folder named after the input audio file (without extension)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    sep_dir = os.path.join(demucs_model_dir, base_name)

    # Define expected output file paths
    vocals_path = os.path.join(sep_dir, "vocals.wav")
    background_path = os.path.join(sep_dir, "no_vocals.wav")
    if not os.path.isfile(vocals_path) or not os.path.isfile(background_path):
        raise RuntimeError("Demucs output files not found")
    # Move or copy the results to interim directory for easier access
    shutil.copy(vocals_path, os.path.join(job_dir, "vocals.wav"))
    shutil.copy(background_path, os.path.join(job_dir, "background.wav"))
    return {
        "vocals": os.path.join(job_dir, "vocals.wav"),
        "background": os.path.join(job_dir, "background.wav"),
    }
