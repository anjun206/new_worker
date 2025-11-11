from typing import Any, Dict, Optional
from app.config import JobProcessingError
import os, logging
from app.config import DEFAULT_TARGET_LANG, DEFAULT_SOURCE_LANG, LOG_LEVEL
from ..status import post_status
from app.services import run_asr

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


class FullPipeline:
    def __init__(self, payload: Dict[str, Any]):
        self.job_id = payload.get("job_id")
        self.project_id = payload.get("project_id")
        self.input_key = payload.get("input_key")
        self.callback_url = payload.get("callback_url")
        self.__validation_check()
        
        self.target_lang = payload.get("target_lang") or DEFAULT_TARGET_LANG
        self.source_lang = payload.get("source_lang") or DEFAULT_SOURCE_LANG
        self.workdir = self.__ensure_workdir(self.job_id)
        self.extension = os.path.splitext(self.input_key)[1]
        self.local_input = os.path.join(self.workdir, f"input{self.extension or '.mp4'}")

    def process(self):
        try:
            logger.info("Downloading s3://%s/%s to %s", self.bucket, self.input_key, self.local_input)
            self.s3_client.download_file(self.bucket, self.input_key, self.local_input)

            post_status(
                self.callback_url, 
                "in_progress", 
                stage_status="processing",
                project_id=self.project_id,
                metadata={
                    "stage": "downloaded",
                    "job_id": self.job_id,
                    "project_id": self.project_id,
                },
            )

            meta = run_asr(self.job_id, self.local_input)

            self.__post_status(
                self.callback_url, "in_progress", metadata={"stage": "stt_completed"}
            )
            self.__post_status(
                self.callback_url, "in_progress", metadata={"stage": "mt_prepare"}
            )
            translations = translate_stage(
                meta["segments"], src=source_lang, tgt=target_lang
            )
            meta["translations"] = translations
            save_meta(workdir, meta)
            self.__post_status(
                callback_url, "in_progress", metadata={"stage": "mt_completed"}
            )
            self.__post_status(
                callback_url,
                "in_progress",
                stage_id="tts",
                stage_status="processing",
                project_id=project_id,
                metadata={
                    "stage": "tts_prepare",
                    "job_id": job_id,
                    "project_id": project_id,
                },
            )

            asyncio.run(tts_finalize_stage(job_id, target_lang, None))
            output_path = mux_stage(job_id)
            meta = load_meta(workdir)
            self._prepare_segment_assets(project_id, job_id, meta)
            meta = load_meta(workdir)
            segments_payload = _build_segment_payload(meta, translations)
            result_key = self.result_video_prefix.format(
                project_id=project_id, job_id=job_id
            )
            metadata_key = self.result_meta_prefix.format(
                project_id=project_id, job_id=job_id
            )
            logger.info("Uploading result video to s3://%s/%s", self.bucket, result_key)
            self.s3.upload_file(output_path, self.bucket, result_key)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=metadata_key,
                Body=json.dumps(
                    {
                        "job_id": job_id,
                        "project_id": project_id,
                        "segments": segments_payload,
                        "target_lang": target_lang,
                        "source_lang": source_lang,
                        "input_key": input_key,
                        "result_key": result_key,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                ContentType="application/json",
            )
            status_payload = {
                "stage": "tts_completed",
                "segments_count": len(segments_payload),
                "segments": segments_payload,
                "metadata_key": metadata_key,
                "result_key": result_key,
                "target_lang": target_lang,
                "source_lang": source_lang,
                "input_key": input_key,
                "segment_assets_prefix": self.interim_segment_prefix.format(
                    project_id=project_id, job_id=job_id
                ),
            }
            self.__post_status(
                callback_url,
                "done",
                stage_id="mux",
                stage_status="done",
                project_id=project_id,
                result_key=result_key,
                metadata=status_payload,
            )
        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(callback_url, wrapped)
            raise wrapped

    def __ensure_workdir(self, job_id: str) -> str:
        workdir = os.path.join("/app/data", job_id)
        os.makedirs(workdir, exist_ok=True)
        return workdir
    
    def __validation_check(self):
        if not all([self.job_id, self.project_id, self.input_key, self.callback_url]):
            raise JobProcessingError("Missing required job fields in payload")