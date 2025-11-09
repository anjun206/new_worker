# mux.py
import os
import subprocess
from pydub import AudioSegment


def mux_audio_video(job_id: str):
    """Combine synthesized speech with background audio and mux with original video."""
    job_dir = os.path.join("interim", job_id)
    background_path = os.path.join(job_dir, "background.wav")
    tts_dir = os.path.join(job_dir, "tts_audio")
    video_input = None
    # Find which media_id was used for this job (by looking up the original audio path saved)
    audio_path = os.path.join(job_dir, "audio.wav")
    if os.path.isfile(audio_path):
        # Derive input video path from stored audio (if we saved a reference in transcript perhaps)
        # Assuming a simple mapping: inputs/<media_id>/source.mp4
        # We might store media_id in a file at interim if needed. For now, search inputs for matching audio file name.
        base = os.path.basename(audio_path)
        media_id = None
        for d in os.listdir("inputs"):
            if base in os.listdir(os.path.join("inputs", d)):
                media_id = d
                break
        if media_id:
            video_input = os.path.join("inputs", media_id, "source.mp4")
    if not video_input or not os.path.isfile(video_input):
        raise RuntimeError("Original video file not found for muxing.")
    if not os.path.isfile(background_path):
        raise FileNotFoundError("Background audio not found. Run Demucs stage.")
    if not os.path.isdir(tts_dir):
        raise FileNotFoundError("TTS audio segments not found. Run TTS stage.")

    # Load background audio
    background_audio = AudioSegment.from_wav(background_path)
    total_duration_ms = len(background_audio)
    # Start with background as the base for final mix
    final_audio = background_audio[:]  # copy

    # Overlay each synthesized speech segment at the correct position
    for fname in os.listdir(tts_dir):
        if fname.endswith(".wav"):
            segment_audio = AudioSegment.from_wav(os.path.join(tts_dir, fname))
            # Parse start time from filename (we included start time in name for convenience)
            try:
                start_time = float(os.path.splitext(fname)[0].split("_")[-1])
            except:
                start_time = 0.0
            start_ms = int(start_time * 1000)
            # If segment audio is longer than original duration segment, we can truncate or let it overlap slightly
            # Ensure not to exceed total duration
            if start_ms < 0:
                start_ms = 0
            # Overlay (mix) the voice segment onto the final audio at the given start time
            final_audio = final_audio.overlay(segment_audio, position=start_ms)

    # Ensure the final audio length matches the original background audio length (pad if needed)
    if len(final_audio) < total_duration_ms:
        silence = AudioSegment.silent(duration=(total_duration_ms - len(final_audio)))
        final_audio = final_audio + silence
    elif len(final_audio) > total_duration_ms:
        final_audio = final_audio[:total_duration_ms]

    # Export the mixed audio to outputs directory
    output_dir = os.path.join("outputs", job_id)
    os.makedirs(output_dir, exist_ok=True)
    final_audio_path = os.path.join(output_dir, "dubbed_audio.wav")
    final_audio.export(final_audio_path, format="wav")

    # Combine the new audio with the original video (without its audio)
    output_video_path = os.path.join(output_dir, "dubbed_video.mp4")
    # Use ffmpeg to mux: replace audio in original video with final_audio
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_input,
        "-i",
        final_audio_path,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_video_path,
    ]
    subprocess.run(cmd, check=True)
    return {"output_video": output_video_path, "output_audio": final_audio_path}
