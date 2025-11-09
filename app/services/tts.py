# tts.py
import os
import json
from pydub import AudioSegment

# Attempt to import CosyVoice (assuming installation)
try:
    import cosyvoice

    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False


def generate_tts(job_id: str, target_lang: str):
    """Generate speech audio for translated text segments using CosyVoice (if available)."""
    job_dir = os.path.join("interim", job_id)
    trans_path = os.path.join(job_dir, "translated.json")
    if not os.path.isfile(trans_path):
        raise FileNotFoundError(
            "Translated text not found. Run translation stage first."
        )
    with open(trans_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Ensure we have the vocals audio (for voice reference and timing reference)
    vocals_path = os.path.join(job_dir, "vocals.wav")
    if not os.path.isfile(vocals_path):
        raise FileNotFoundError("Vocals track not found. Run Demucs stage first.")
    vocals_audio = AudioSegment.from_wav(vocals_path)

    # Prepare a reference voice sample for each speaker from the original vocals
    speaker_samples = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in speaker_samples:
            seg_duration = seg["end"] - seg["start"]
            # Convert times to milliseconds for pydub
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            # Extract the segment audio from the vocals track as reference sample
            sample_audio = vocals_audio[start_ms:end_ms]
            duration_ms = len(sample_audio)
            # If the segment is very short, we may combine subsequent segments or handle accordingly.
            # For simplicity, we use the first occurrence (or longest segment could be chosen).
            speaker_samples[spk] = sample_audio
    # (Optionally, could refine by choosing the longest segment per speaker for better quality)

    # Directory to store synthesized voice segments
    tts_dir = os.path.join(job_dir, "tts_audio")
    os.makedirs(tts_dir, exist_ok=True)
    synthesized_segments = []

    for seg in segments:
        spk = seg["speaker"]
        text = seg.get("translation", seg.get("text", ""))
        segment_start = seg["start"]
        segment_end = seg["end"]
        duration = segment_end - segment_start  # in seconds
        output_file = os.path.join(tts_dir, f"{spk}_{segment_start:.2f}.wav")

        if COSYVOICE_AVAILABLE:
            # Pseudo-code for CosyVoice usage – assuming CosyVoice provides a TTS interface
            # Initialize or load CosyVoice model (if not already globally loaded)
            # e.g., tts_engine = cosyvoice.TTS(model="CosyVoice2-0.5B", device="cuda")
            # Use reference audio from speaker_samples[spk] to clone voice
            ref_wav_path = os.path.join(tts_dir, f"{spk}_ref.wav")
            # Save reference sample to file (CosyVoice might need a file path)
            speaker_samples[spk].export(ref_wav_path, format="wav")
            try:
                # Synthesize speech using CosyVoice
                # (Note: Actual CosyVoice API usage depends on its library interface)
                synthesized_wav = None
                # Example placeholder call (to be replaced with actual CosyVoice inference):
                # synthesized_wav = cosyvoice.synthesize(text, reference_audio=ref_wav_path, language=target_lang)
                # Save the synthesized_wav (if cosyvoice returns raw audio or file path)
                if synthesized_wav:
                    AudioSegment(
                        synthesized_wav, sample_width=2, frame_rate=24000, channels=1
                    ).export(output_file, format="wav")
                else:
                    raise RuntimeError(
                        "CosyVoice synthesis not implemented in prototype."
                    )
            except Exception as e:
                raise RuntimeError(f"CosyVoice TTS failed for segment {seg}: {e}")
        else:
            # Fallback: use a simple TTS (e.g., gTTS or pyttsx3) for demonstration (monotonic voice, no cloning)
            try:
                from gtts import gTTS

                tts = gTTS(text=text, lang=target_lang)
                tts.save(output_file)
            except Exception:
                # If gTTS not available or fails, create a silent or dummy audio of correct duration
                silence = AudioSegment.silent(duration=int(duration * 1000))
                silence.export(output_file, format="wav")

        synthesized_segments.append(
            {
                "speaker": spk,
                "start": segment_start,
                "end": segment_end,
                "audio_file": output_file,
            }
        )
    return synthesized_segments
