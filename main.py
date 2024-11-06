import os
import json
import time
import asyncio
import tempfile
import aiohttp
import base64
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from google.cloud import speech
from google.cloud import texttospeech
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydanticモデル
class AnswerSubmission(BaseModel):
    answer: str
    questionIndex: int

class MusicGeneration(BaseModel):
    answers: List[str]

class TextToSpeech:
    def __init__(self, client):
        self.client = client
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name="ja-JP-Neural2-B",
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

    async def synthesize_speech(self, text: str) -> str:
        """
        テキストを音声に変換しBase64エンコードされた文字列を返す
        """
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            return base64.b64encode(response.audio_content).decode()
        except Exception as e:
            logger.error(f"Speech synthesis error: {str(e)}")
            logger.exception(e)
            raise

def setup_google_credentials():
    """Setup Google Cloud credentials from base64 encoded string"""
    try:
        encoded_credentials = os.getenv('GOOGLE_CREDENTIALS_BASE64')
        if not encoded_credentials:
            raise ValueError("GOOGLE_CREDENTIALS_BASE64 environment variable is not set")
        
        credentials_json = base64.b64decode(encoded_credentials).decode('utf-8')
        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            temp_file.write(credentials_json.encode())
            temp_file.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
            return temp_file.name
        except Exception as e:
            os.unlink(temp_file.name)
            raise e
    except Exception as e:
        logger.error(f"Failed to setup Google credentials: {e}")
        raise

# アプリケーションの状態を保持
app_state: Dict = {
    "answers": {},
    "current_progress": 0,
    "generation_status": None,
    "tts_client": None,
    "speech_client": None,
    "tts": None
}

# FastAPI initialization with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: setup credentials and create client session
    credentials_path = None
    try:
        credentials_path = setup_google_credentials()
        app.state.client_session = aiohttp.ClientSession()
        
        # Google Cloud Clientsの初期化
        app_state["tts_client"] = texttospeech.TextToSpeechClient()
        app_state["speech_client"] = speech.SpeechClient()
        
        # TTSの初期化
        app_state["tts"] = TextToSpeech(app_state["tts_client"])
        
        logger.info("Initialized application services")
        
        yield
        
    finally:
        # Cleanup
        if app.state.client_session:
            await app.state.client_session.close()
        if credentials_path and os.path.exists(credentials_path):
            try:
                os.unlink(credentials_path)
            except Exception as e:
                logger.error(f"Failed to delete credentials file: {e}")
        logger.info("Cleanup completed")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-song-creator-web.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 質問リスト
QUESTIONS = [
    "あなたの青春時代を一言で表すと？",
    "その時期にあなたが最も夢中になっていたものは？",
    "青春時代の挫折や失敗を乗り越えた時の気持ちを一言で？",
    "その頃のあなたにとって最も大切だったものは？",
    "今、あの頃の自分に伝えたい言葉を一つ挙げるとしたら？"
]

# エンドポイント：インタビュー開始
@app.post("/start-interview")
async def start_interview():
    try:
        if not app_state["tts"]:
            raise HTTPException(
                status_code=500,
                detail="Text-to-Speech service not initialized"
            )

        greeting = "こんにちは！青春ソングを作るためのインタビューを始めましょう。各質問に一言で答えてください。"
        audio_base64 = await app_state["tts"].synthesize_speech(greeting)
        
        # 状態をリセット
        app_state["answers"] = {}
        app_state["current_progress"] = 0
        app_state["generation_status"] = None
        
        return {
            "text": greeting,
            "audio": audio_base64
        }
    except Exception as e:
        logger.error(f"Error in start_interview: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# エンドポイント：質問を取得
@app.post("/get-question")
async def get_question(data: dict):
    try:
        index = data.get("index", 0)
        if index < 0 or index >= len(QUESTIONS):
            raise HTTPException(status_code=400, detail="Invalid question index")

        question = QUESTIONS[index]
        audio_base64 = await app_state["tts"].synthesize_speech(question)
        
        return {
            "text": question,
            "audio": audio_base64
        }
    except Exception as e:
        logger.error(f"Error in get_question: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    
# エンドポイント：回答を送信
@app.post("/submit-answer")
async def submit_answer(data: AnswerSubmission):
    try:
        # 回答を保存
        app_state["answers"][data.questionIndex] = data.answer
        logger.info(f"Saved answer for question {data.questionIndex}: {data.answer}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in submit_answer: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# エンドポイント：音声認識
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_audio_path = None
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio_path = temp_audio.name
            content = await file.read()
            temp_audio.write(content)

        # 音声ファイルを読み込み
        with open(temp_audio_path, "rb") as audio_file:
            content = audio_file.read()

        # 音声認識の設定
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="ja-JP",
        )

        # 音声認識を実行
        response = app_state["speech_client"].recognize(config=config, audio=audio)
        transcription = response.results[0].alternatives[0].transcript if response.results else ""
        
        logger.info(f"Transcribed text: {transcription}")
        return {"transcription": transcription}

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 一時ファイルの削除
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

# 楽曲生成の実処理
async def generate_music_with_suno(themes: List[str]) -> dict:
    """Generate music using Suno AI"""
    try:
        suno_api_url = "https://api.goapi.ai/api/suno/v1/music"
        
        # Suno APIの認証情報を設定
        headers = {
            "X-API-Key": os.getenv('SUNO_API_KEY'),
            "Content-Type": "application/json"
        }
        
        # プロンプトの生成
        themes_text = " ".join(themes)
        description = f"Simple J-pop about {themes_text}. No repeats. Female JP vocal."
        
        if len(description) > 200:
            max_themes_length = 100
            if len(themes_text) > max_themes_length:
                themes_text = themes_text[:max_themes_length] + "..."
                description = f"Create a nostalgic J-pop song about {themes_text}. Female JP vocal."
        
        logger.info(f"Generating music with prompt: {description}")
        
        # APIリクエストの設定
        payload = {
            "custom_mode": False,
            "mv": "chirp-v3-5",
            "input": {
                "gpt_description_prompt": description,
                "make_instrumental": False,
                "voice": "female",
                "style": "j-pop",
                "temperature": 0.1,
                "top_k": 5,
                "top_p": 0.3,
                "voice_settings": {
                    "gender": "female",
                    "language": "japanese",
                    "style": "clear",
                    "variation": "single"
                }
            }
        }

        # タスク作成
        async with app.state.client_session.post(suno_api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Suno API Error: Status {response.status}")
                logger.error(f"Response: {error_text}")
                raise HTTPException(status_code=response.status, detail=f"Suno API Error: {error_text}")
            
            task_data = await response.json()
            task_id = task_data['data']['task_id']
            logger.info(f"Task ID: {task_id}")

        # 生成状態の確認
        while True:
            async with app.state.client_session.get(f"{suno_api_url}/{task_id}", headers=headers) as status_response:
                status_data = await status_response.json()
                
                if 'data' not in status_data:
                    logger.error("Invalid response structure")
                    raise HTTPException(status_code=500, detail="Invalid response format")

                status = status_data['data'].get('status')
                
                if status == 'completed':
                    # 生成完了時の処理
                    video_url = None
                    if 'output' in status_data['data']:
                        video_url = status_data['data']['output'].get('video_url')
                    elif 'clips' in status_data['data']:
                        clips = status_data['data']['clips']
                        if clips and isinstance(clips, dict):
                            first_clip = next(iter(clips.values()))
                            video_url = first_clip.get('video_url')
                    else:
                        video_url = status_data['data'].get('video_url')

                    if video_url:
                        logger.info(f"Music generation completed with URL: {video_url}")
                        return {"status": "success", "video_url": video_url}
                    raise HTTPException(status_code=500, detail="No video URL found in completed response")

                elif status == 'failed':
                    error_message = status_data['data'].get('error', 'Unknown error')
                    logger.error(f"Generation failed: {error_message}")
                    raise HTTPException(status_code=500, detail=error_message)

                elif status == 'processing':
                    progress = status_data['data'].get('progress', 0)
                    app_state["current_progress"] = progress
                    logger.info(f"Generation progress: {progress}%")

            await asyncio.sleep(3)  # 3秒待機してから次のチェック
            
    except Exception as e:
        logger.error(f"Error in music generation: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# エンドポイント：楽曲生成
@app.post("/generate-music")
async def generate_music(data: MusicGeneration):
    try:
        if not data.answers:
            raise HTTPException(status_code=400, detail="No answers provided")

        # 生成状態の初期化
        app_state["generation_status"] = "generating"
        app_state["current_progress"] = 0

        # 楽曲生成の実行
        result = await generate_music_with_suno(data.answers)
        
        # 生成完了
        app_state["generation_status"] = "completed"
        return result

    except Exception as e:
        # エラー発生時の処理
        app_state["generation_status"] = "failed"
        logger.error(f"Error in generate_music: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# エンドポイント：生成進捗状況を取得
@app.get("/generation-progress")
async def get_generation_progress():
    try:
        return {
            "progress": app_state["current_progress"],
            "status": app_state["generation_status"]
        }
    except Exception as e:
        logger.error(f"Error in get_generation_progress: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# メインの実行（開発環境用）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
