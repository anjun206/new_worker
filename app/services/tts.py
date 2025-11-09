# tts.py
import os
import json
from pydub import AudioSegment

# CosyVoice가 설치되어 있으면 사용
try:
    import cosyvoice

    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False


def generate_tts(job_id: str, target_lang: str):
    """번역된 구간을 CosyVoice(가능한 경우) 또는 대체 TTS로 음성 합성합니다."""
    job_dir = os.path.join("interim", job_id)
    trans_path = os.path.join(job_dir, "translated.json")
    if not os.path.isfile(trans_path):
        raise FileNotFoundError(
            "Translated text not found. Run translation stage first."
        )
    with open(trans_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 음성 특징 추출과 길이 정렬을 위해 보컬 트랙이 필요
    vocals_path = os.path.join(job_dir, "vocals.wav")
    if not os.path.isfile(vocals_path):
        raise FileNotFoundError("Vocals track not found. Run Demucs stage first.")
    vocals_audio = AudioSegment.from_wav(vocals_path)

    # 각 화자별로 원본 보컬에서 참조 음성 샘플을 추출
    speaker_samples = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in speaker_samples:
            seg_duration = seg["end"] - seg["start"]
            # pydub 사용을 위해 밀리초 단위로 변환
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            # 보컬 트랙에서 구간을 잘라 참조 음성으로 사용
            sample_audio = vocals_audio[start_ms:end_ms]
            duration_ms = len(sample_audio)
            # 구간이 짧다면 이어지는 구간을 결합할 수도 있지만 현재는 최초 구간을 사용
            speaker_samples[spk] = sample_audio
    # 필요하다면 가장 긴 구간을 고르는 방식으로 품질을 높일 수 있음

    # 합성된 음성 세그먼트를 저장할 디렉터리
    tts_dir = os.path.join(job_dir, "tts_audio")
    os.makedirs(tts_dir, exist_ok=True)
    synthesized_segments = []

    for seg in segments:
        spk = seg["speaker"]
        text = seg.get("translation", seg.get("text", ""))
        segment_start = seg["start"]
        segment_end = seg["end"]
        duration = segment_end - segment_start  # 초 단위 지속시간
        output_file = os.path.join(tts_dir, f"{spk}_{segment_start:.2f}.wav")

        if COSYVOICE_AVAILABLE:
            # CosyVoice 사용 시 필요한 초기화/추론 절차 (실제 API에 맞게 수정 필요)
            # 예: tts_engine = cosyvoice.TTS(model="CosyVoice2-0.5B", device="cuda")
            # 참조 음성 파일을 만들어 화자 음색을 전달
            ref_wav_path = os.path.join(tts_dir, f"{spk}_ref.wav")
            # CosyVoice가 파일 입력을 요구할 수 있으므로 참조 샘플을 저장
            speaker_samples[spk].export(ref_wav_path, format="wav")
            try:
                # CosyVoice로 음성을 합성 (구체적인 사용법은 라이브러리에 따라 다름)
                synthesized_wav = None
                # 예시: cosyvoice.synthesize(text, reference_audio=ref_wav_path, language=target_lang)
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
            # CosyVoice가 없으면 gTTS 등 간단한 TTS로 대체 (음색 클로닝 없음)
            try:
                from gtts import gTTS

                tts = gTTS(text=text, lang=target_lang)
                tts.save(output_file)
            except Exception:
                # gTTS 사용이 어렵다면 길이만 맞춘 무음 파일 생성
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
