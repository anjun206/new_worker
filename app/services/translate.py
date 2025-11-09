# translate.py
import os
import json

# Placeholder: using googletrans for example (this requires internet, so in practice replace with official API)
from googletrans import Translator

translator = Translator()


def translate_transcript(job_id: str, target_lang: str):
    """Translate the transcribed text segments to the target language."""
    job_dir = os.path.join("interim", job_id)
    transcript_path = os.path.join(job_dir, "transcript.json")
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError("Transcript not found. Run ASR stage first.")
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    translated_segments = []
    for seg in segments:
        text = seg["text"]
        # Use a translation API or library to translate text
        try:
            # Using googletrans as a placeholder; for actual use, integrate Gemini API call here.
            result = translator.translate(text, dest=target_lang)
            translated_text = result.text
        except Exception as e:
            # Fallback: if translation fails, just copy original text (or handle accordingly)
            translated_text = text
        new_seg = seg.copy()
        new_seg["translation"] = translated_text
        translated_segments.append(new_seg)

    # Save translated segments
    trans_out_path = os.path.join(job_dir, "translated.json")
    with open(trans_out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    return translated_segments
