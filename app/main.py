# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uuid
import os

# Import our pipeline stage functions
from stt import run_asr
from demucs_split import split_vocals
from translate import translate_transcript
from tts import generate_tts
from mux import mux_audio_video


# Define request/response models for documentation
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
    description="End-to-end video dubbing pipeline API",
)

# Helper: ensure base folders exist
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
    Upload a video (or specify media_id of an existing input video) to transcribe audio with WhisperX.
    Returns a new job_id and the transcription segments with speaker labels.
    """
    # If file is uploaded, save it under inputs/ with a new media_id (or provided media_id)
    if file:
        if media_id is None:
            media_id = str(uuid.uuid4())  # generate a new media_id if not provided
        media_dir = os.path.join("inputs", media_id)
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, "source.mp4")
        with open(file_path, "wb") as f:
            f.write(await file.read())
    else:
        # If no file uploaded, media_id must refer to an existing video file
        if media_id is None:
            return JSONResponse(status_code=400, content={"error": "No media provided"})
        file_path = os.path.join("inputs", media_id, "source.mp4")
        if not os.path.isfile(file_path):
            return JSONResponse(
                status_code=404, content={"error": f"Media {media_id} not found"}
            )

    # Create a new job ID for this transcription task
    job_id = str(uuid.uuid4())
    try:
        segments = run_asr(media_id, job_id, hf_token)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "segments": segments}


@app.post("/demucs")
async def demucs_endpoint(job_id: str):
    """
    Separate vocals from background in the audio of the given job.
    """
    try:
        result = split_vocals(job_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"message": "Vocals and background separated", "files": result}


@app.post("/translate")
async def translate_endpoint(request: TranslateRequest):
    """
    Translate the transcribed text for the given job to the target language.
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
    Generate translated speech audio for each segment of the given job using TTS.
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
    Mix the synthesized voice with background and mux with original video to produce the dubbed video.
    Returns the final video file.
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
    # Return the output video file as a response for download
    return FileResponse(
        output_video, media_type="video/mp4", filename=f"dubbed_{job_id}.mp4"
    )
