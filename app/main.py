# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uuid
import os

# 파이프라인 각 단계를 담당하는 함수 불러오기
from services.stt import run_asr
from services.demucs_split import split_vocals
from services.translate import translate_transcript
from services.tts import generate_tts
from services.mux import mux_audio_video


# 문서화를 위한 요청/응답 모델 정의
class ASRResponse(BaseModel):
    job_id: str
    segments: list


class TranslateRequest(BaseModel):
    job_id: str
    target_lang: str


class TTSRequest(BaseModel):
    job_id: str
    target_lang: str


app = FastAPI(
    docs_url="/",
    title="Video Dubbing API",
    description="엔드 투 엔드 비디오 더빙 파이프라인 API",
)

# 기본 작업 폴더가 없으면 생성
os.makedirs("inputs", exist_ok=True)
os.makedirs("interim", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(
    media_id: str = Form(None),
    file: UploadFile = File(None),
    hf_token: str = Form(None),
):
    """
    새 영상을 업로드하거나 기존 media_id를 지정해 WhisperX로 음성을 추출합니다.
    새 job_id와 화자 정보가 포함된 전사 구간 목록을 반환합니다.
    """
    # 업로드된 파일이 있으면 inputs/<media_id>/source.mp4 형태로 저장
    if file:
        if media_id is None:
            media_id = str(uuid.uuid4())  # media_id가 없으면 새로 생성
        media_dir = os.path.join("inputs", media_id)
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, "source.mp4")
        with open(file_path, "wb") as f:
            f.write(await file.read())
    else:
        # 업로드 파일이 없다면 media_id가 기존 입력 영상을 가리켜야 함
        if media_id is None:
            return JSONResponse(status_code=400, content={"error": "No media provided"})
        file_path = os.path.join("inputs", media_id, "source.mp4")
        if not os.path.isfile(file_path):
            return JSONResponse(
                status_code=404, content={"error": f"Media {media_id} not found"}
            )

    # 작업 단위 식별을 위한 job_id 생성
    job_id = str(uuid.uuid4())
    try:
        segments = run_asr(media_id, job_id, hf_token)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "segments": segments}


@app.post("/demucs")
async def demucs_endpoint(job_id: str):
    """
    지정된 job_id의 오디오에서 보컬과 배경음을 분리합니다.
    """
    try:
        result = split_vocals(job_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"message": "Vocals and background separated", "files": result}


@app.post("/translate")
async def translate_endpoint(request: TranslateRequest):
    """
    지정된 job_id의 전사 텍스트를 target_lang으로 번역합니다.
    """
    job_id = request.job_id
    target_lang = request.target_lang
    try:
        segments = translate_transcript(job_id, target_lang)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {
        "job_id": job_id,
        "target_lang": target_lang,
        "translated_segments": segments,
    }


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    """
    지정된 job_id에 대해 각 구간의 번역된 음성을 합성합니다.
    """
    job_id = request.job_id
    target_lang = request.target_lang
    try:
        segments = generate_tts(job_id, target_lang)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "audio_segments": segments}


@app.post("/mux")
async def mux_endpoint(job_id: str):
    """
    합성된 음성과 배경음을 섞어 원본 영상과 결합해 더빙 영상을 생성합니다.
    최종 mp4 파일을 반환합니다.
    """
    try:
        paths = mux_audio_video(job_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    output_video = paths["output_video"]
    if not os.path.isfile(output_video):
        return JSONResponse(
            status_code=500, content={"error": "Muxing failed, output video not found"}
        )
    # 생성된 비디오 파일을 바로 다운로드할 수 있도록 응답으로 반환
    return FileResponse(
        output_video, media_type="video/mp4", filename=f"dubbed_{job_id}.mp4"
    )
