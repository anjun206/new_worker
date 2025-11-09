# translate.py
import os
import json

# 예시용으로 googletrans 사용 (실제 서비스에서는 공식 번역 API로 교체 필요)
from googletrans import Translator

translator = Translator()


def translate_transcript(job_id: str, target_lang: str):
    """전사된 구간 텍스트를 지정한 언어로 번역합니다."""
    job_dir = os.path.join("interim", job_id)
    transcript_path = os.path.join(job_dir, "transcript.json")
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError("Transcript not found. Run ASR stage first.")
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    translated_segments = []
    for seg in segments:
        text = seg["text"]
        # 번역 API/라이브러리를 사용해 텍스트 변환
        try:
            # 현재는 googletrans를 예제로 사용하며, 실서비스에서는 Gemini 등의 API로 대체
            result = translator.translate(text, dest=target_lang)
            translated_text = result.text
        except Exception as e:
            # 실패 시에는 원문을 그대로 사용하거나 별도 예외 처리를 수행
            translated_text = text
        new_seg = seg.copy()
        new_seg["translation"] = translated_text
        translated_segments.append(new_seg)

    # 번역 결과 저장
    trans_out_path = os.path.join(job_dir, "translated.json")
    with open(trans_out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    return translated_segments
