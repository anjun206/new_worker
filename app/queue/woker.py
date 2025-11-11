import os
from app.config import JOB_QUEUE_URL, AWS_REGION, AWS_S3_BUCKET, DEFAULT_TARGET_LANG, DEFAULT_SOURCE_LANG, VISIBILITY_TIMEOUT, POLL_WAIT, CALLBACK_LOCALHOST_HOST, PROFILE, LOG_LEVEL
from botocore.exceptions import BotoCoreError, ClientError
import logging, time, requests, boto3, json
from app.config import JobProcessingError
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

class QueueWorker:
    def __init__(self):
        self.queue_url = JOB_QUEUE_URL
        self.bucket = AWS_S3_BUCKET
        self.default_target_lang = DEFAULT_TARGET_LANG
        self.default_source_lang = DEFAULT_SOURCE_LANG
        self.visibility_timeout = VISIBILITY_TIMEOUT
        self.poll_wait = POLL_WAIT
        self.callback_localhost_host = CALLBACK_LOCALHOST_HOST
        
        self.result_video_prefix = os.getenv(
            "JOB_RESULT_VIDEO_PREFIX",
            "projects/{project_id}/outputs/videos/{job_id}.mp4",
        )
        self.result_meta_prefix = os.getenv(
            "JOB_RESULT_METADATA_PREFIX",
            "projects/{project_id}/outputs/metadata/{job_id}.json",
        )
        self.interim_segment_prefix = os.getenv(
            "JOB_INTERIM_SEGMENT_PREFIX",
            "projects/{project_id}/interim/{job_id}/segments",
        )
        session_kwargs: dict = {}
        if PROFILE:
            session_kwargs["profile_name"] = PROFILE
        boto_session = boto3.Session(region_name=AWS_REGION, **session_kwargs)
        self.sqs_client = boto_session.client("sqs", region_name=AWS_REGION)
        self.s3_client = boto_session.client("s3", region_name=AWS_REGION)
        self.http = requests.Session()

    def poll_forever(self):
        while True:
            try:
                messages = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=self.poll_wait,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
            except (BotoCoreError, ClientError) as exc:
                logger.error("Failed to poll SQS: %s", exc)
                time.sleep(5)
                continue

            for message in messages.get("Messages", []):
                receipt = message["ReceiptHandle"]
                rc = int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
                logger.info("SQS received (ReceiveCount=%s, MessageId=%s)", rc, message.get("MessageId"))

                try:
                    body = json.loads(message.get("Body", "{}"))
                except json.JSONDecodeError:
                    logger.error("Invalid message body, deleting: %s", message.get("Body"))
                    self._delete_message(receipt)
                    continue

                success = False
                try:
                    self.__handle_job(body)
                    success = True
                except JobProcessingError as exc:
                    logger.error("Job %s failed: %s", body.get("job_id"), exc)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Unexpected error handling message: %s", exc)

                finally:
                    self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)
                    logger.info("SQS deleted: %s", message.get("MessageId"))

    def __handle_job(self, payload: Dict[str, Any]) -> None:
        task = (payload.get("task") or "full_pipeline").lower()

        if task == "full_pipeline":  # preTTS
            self.__handle_full_pipeline(payload)
        elif task == "segment_tts":
            self._handle_segment_tts(payload)
        elif task in {"segment_mix", "tts_bgm_mix"}:
            self._handle_segment_mix(payload)
            # 최종 TTS + mix -> 최종 결과물 나오는 ~ handle_최종 어쩌구 저쩌구(payload) 필요
        else:
            raise JobProcessingError(f"Unsupported task type: {task}")

    
    def __handle_full_pipeline(self, payload: Dict[str, Any]):
        job_id = payload.get("job_id")
        project_id = payload.get("project_id")
        input_key = payload.get("input_key")
        callback_url = payload.get("callback_url")
        if not all([job_id, project_id, input_key, callback_url]):
            raise JobProcessingError("Missing required job fields in payload")
        
        target_lang = payload.get("target_lang") or self.default_target_lang
        source_lang = payload.get("source_lang") or self.default_source_lang
        workdir = self.__ensure_workdir(job_id)
        extension = os.path.splitext(input_key)[1]
        local_input = os.path.join(workdir, f"input{extension or '.mp4'}")

        try:
            logger.info("Downloading s3://%s/%s to %s", self.bucket, input_key, local_input)
            self.s3_client.download_file(self.bucket, input_key, local_input)
            self.__post_status(
                callback_url,
                "in_progress",
                stage_id="asr",
                stage_status="processing",
                project_id=project_id,
                metadata={
                    "stage": "downloaded",
                    "job_id": job_id,
                    "project_id": project_id,
                },
            )

            meta = _run_asr_stage(job_id, local_input)
            self.__post_status(
                callback_url, "in_progress", metadata={"stage": "stt_completed"}
            )
            self.__post_status(
                callback_url, "in_progress", metadata={"stage": "mt_prepare"}
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
        
    def ___post_status(
        self,
        callback_url: str,
        status: str,
        *,
        result_key: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stage_id: Optional[str] = None,
        stage_status: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        stage_id = stage_id or "pipeline"
        stage_status = stage_status or ("done" if status == "done" else "processing")

        payload: Dict[str, Any] = {
            "status": status,
            "stage_id": stage_id,
            "stage_status": stage_status,
        }
        if project_id is not None:
            payload["project_id"] = project_id
        if result_key is not None:
            payload["result_key"] = result_key
        if error is not None:
            payload["error"] = error
        if metadata is not None:
            payload["metadata"] = metadata

        target_url = self._normalize_callback_url(callback_url)

        try:
            resp = self.http.post(target_url, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise JobProcessingError(f"Callback request failed: {exc}") from exc

        if not resp.ok:
            raise JobProcessingError(
                f"Callback responded with {resp.status_code}: {resp.text[:200]}"
            )







if __name__ == "__main__":
    worker = QueueWorker()
    worker.poll_forever()
