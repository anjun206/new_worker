# stt.py
import os
import json
import subprocess
import whisperx

from whisperx import DiarizationPipeline


def run_asr(media_id: str, job_id: str, hf_token: str = None):
    """Transcribe the input video using WhisperX and perform speaker diarization."""
    # File paths
    input_video = os.path.join("inputs", media_id, "source.mp4")
    job_dir = os.path.join("interim", job_id)
    os.makedirs(job_dir, exist_ok=True)
    audio_path = os.path.join(job_dir, "audio.wav")

    # 1. Extract audio from video (mono, 16kHz for Whisper)
    # Using ffmpeg via subprocess (ffmpeg must be installed)
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

    # 2. Load WhisperX model for transcription
    device = "cuda" if whisperx.utils.get_device() == "cuda" else "cpu"
    model = whisperx.load_model(
        "large-v2", device=device
    )  # using large model for accuracy

    # 3. Transcribe audio to get segments (without word-level alignment yet)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)
    segments = result["segments"]  # list of segments with text and rough timestamps

    # 4. Load alignment model for precise word timings
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
    segments = result_aligned["segments"]  # segments now have word-level timestamps

    # 5. Perform speaker diarization using pyannote (requires HF token for models)
    if hf_token:
        diarization_pipeline = DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarization_segments = diarization_pipeline(audio_path)
        # Assign speaker labels to each segment
        result_segments = whisperx.assign_word_speakers(
            diarization_segments, result_aligned
        )
        segments = result_segments["segments"]
        # The segments now include a "speaker" label for each segment or word

    # 6. Simplify segments data for output (merge words into full segment text per speaker segment)
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

    # 7. Save transcription segments to interim JSON
    transcript_path = os.path.join(job_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_segments, f, ensure_ascii=False, indent=2)

    return output_segments
