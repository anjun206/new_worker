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
from config import ensure_data_dirs, ensure_job_dirs


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
ensure_data_dirs()


@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(
    job_id: str = Form(None),
    file: UploadFile = File(None),
):
    """
    새 영상을 업로드하거나 기존 job_id를 지정해 WhisperX로 음성을 추출합니다.
    job_id와 화자 정보가 포함된 전사 구간 목록을 반환합니다.
    """
    if file:
        job_id = job_id or str(uuid.uuid4())
        paths = ensure_job_dirs(job_id)
        input_path = paths.input_dir / "source.mp4"
        with open(input_path, "wb") as f:
            f.write(await file.read())
    else:
        if job_id is None:
            return JSONResponse(status_code=400, content={"error": "No media provided"})
        paths = ensure_job_dirs(job_id)
        input_path = paths.input_dir / "source.mp4"
        if not input_path.is_file():
            return JSONResponse(
                status_code=404,
                content={"error": f"Input for job {job_id} not found"},
            )

    try:
        segments = run_asr(job_id)
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
