import os
import json
import boto3
import time
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from aws import AWS

aws = AWS()
load_dotenv()

# --- 1. AWS 클라이언트 및 설정 ---

# 파일을 다운로드할 임시 폴더
DOWNLOAD_DIR = "temp_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Boto3 클라이언트 초기화
sqs_client = boto3.client('sqs', region_name=aws.region)
s3_client = boto3.client('s3', region_name=aws.region)

def main_loop():
    """
    SQS 큐를 계속 확인하고, 메시지가 오면 처리합니다.
    """
    while True:
        try:
            # 1. SQS 큐에서 메시지 받기 (Long Polling)
            #    WaitTimeSeconds=20: 메시지가 올 때까지 최대 20초간 대기 (API 비용 절약)
            print("\n----------------------------------\nWaiting for message (Long Polling)...")
            response = sqs_client.receive_message(
                QueueUrl=aws.queue_url,
                MaxNumberOfMessages=1, # 한 번에 1개만 처리
                WaitTimeSeconds=20
            )

            if "Messages" not in response:
                print("받은 메시지가 없습니다. 다시 폴링합니다...")
                continue # 메시지 없으면 루프 처음으로

            # 2. 받은 메시지가 있는지 확인
            message = response['Messages'][0]
            
            receipt_handle = message['ReceiptHandle'] # 메시지 삭제에 필요한 핸들

            # 3. 메시지 본문(Body) 파싱 (S3 정보 추출)
            try:
                body = json.loads(message['Body'])
                s3_bucket = body['s3Bucket']
                s3_key = body['s3Key']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"❗️ 오류: 메시지 형식이 잘못되었습니다. {e}")
                # 잘못된 메시지는 큐에서 삭제 (무한 루프 방지)
                sqs_client.delete_message(QueueUrl=aws.queue_url, ReceiptHandle=receipt_handle)
                continue

            try:
                file_name = s3_key.split('/')[1]

                path = os.path.join(os.path.abspath(DOWNLOAD_DIR), file_name)
                s3_client.download_file(s3_bucket, s3_key, path)
                print("다운로드 성공.")

            except ClientError as e:
                # S3 관련 에러(권한, 파일 없음 등)를 잡아서 출력
                print(f"❗️ S3 다운로드 오류: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
                # 다운로드 실패 시 메시지를 삭제하지 않고 다음 루프로 넘어감
                continue 
            except Exception as e:
                # 기타 예외 처리
                print(f"❗️ 다운로드 중 예외 발생: {e}")
                continue

            sqs_client.delete_message(
                QueueUrl=aws.queue_url,
                ReceiptHandle=receipt_handle
            )

        except Exception as e:
            print(f"❗️ 심각한 오류 발생: {e}")
            # 알 수 없는 오류 발생 시, 큐 재시도를 위해 5초간 대기
            time.sleep(5)
        finally:
            # 7. 로컬에 다운로드한 임시 파일 삭제
            if 'local_file_path' in locals() and os.path.exists(local_file_path):
                os.remove(local_file_path)
                del local_file_path # 변수 정리

if __name__ == "__main__":
    main_loop()