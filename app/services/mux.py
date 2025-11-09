# mux.py
import os
import subprocess
from pydub import AudioSegment


def mux_audio_video(job_id: str):
    """합성 음성과 배경음을 결합하고 원본 영상에 다시 입혀 최종 영상을 생성합니다."""
    job_dir = os.path.join("interim", job_id)
    background_path = os.path.join(job_dir, "background.wav")
    tts_dir = os.path.join(job_dir, "tts_audio")
    video_input = None
    # 저장된 오디오를 바탕으로 어떤 media_id를 사용했는지 역추적
    audio_path = os.path.join(job_dir, "audio.wav")
    if os.path.isfile(audio_path):
        # inputs/<media_id>/source.mp4 구조를 가정하고 해당 media_id를 탐색
        # 필요 시 job 디렉터리에 media_id 정보를 별도로 저장하는 것이 안전함
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

    # 배경 오디오 로드
    background_audio = AudioSegment.from_wav(background_path)
    total_duration_ms = len(background_audio)
    # 배경음을 복제해 믹싱의 베이스로 사용
    final_audio = background_audio[:]  # 복제본

    # 합성된 음성 구간을 적절한 시작 위치에 오버레이
    for fname in os.listdir(tts_dir):
        if fname.endswith(".wav"):
            segment_audio = AudioSegment.from_wav(os.path.join(tts_dir, fname))
            # 파일명 끝부분에 기록된 시작 시간을 파싱
            try:
                start_time = float(os.path.splitext(fname)[0].split("_")[-1])
            except:
                start_time = 0.0
            start_ms = int(start_time * 1000)
            # 구간이 길더라도 전체 길이를 초과하지 않도록 위치 조정
            if start_ms < 0:
                start_ms = 0
            # 해당 위치에 음성 구간을 오버레이
            final_audio = final_audio.overlay(segment_audio, position=start_ms)

    # 필요 시 패딩/트리밍으로 길이를 배경 오디오와 동일하게 맞춤
    if len(final_audio) < total_duration_ms:
        silence = AudioSegment.silent(duration=(total_duration_ms - len(final_audio)))
        final_audio = final_audio + silence
    elif len(final_audio) > total_duration_ms:
        final_audio = final_audio[:total_duration_ms]

    # 믹싱된 오디오를 outputs 디렉터리에 저장
    output_dir = os.path.join("outputs", job_id)
    os.makedirs(output_dir, exist_ok=True)
    final_audio_path = os.path.join(output_dir, "dubbed_audio.wav")
    final_audio.export(final_audio_path, format="wav")

    # 원본 영상과 새 오디오를 결합
    output_video_path = os.path.join(output_dir, "dubbed_video.mp4")
    # ffmpeg로 영상의 오디오 트랙을 교체
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
