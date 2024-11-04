import os
import json
import time
import asyncio
import tempfile
import aiohttp
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
from google.cloud import speech
from google.cloud import texttospeech
import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_google_credentials():
    """Setup Google Cloud credentials from base64 encoded string"""
    try:
        encoded_credentials = os.getenv('GOOGLE_CREDENTIALS_BASE64')
        if not encoded_credentials:
            raise ValueError("GOOGLE_CREDENTIALS_BASE64 environment variable is not set")
        
        # デコードしてJSONを取得
        credentials_json = base64.b64decode(encoded_credentials).decode('utf-8')
        
        # 一時ファイルを作成
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            temp_file.write(credentials_json.encode())
            temp_file.flush()
            # 環境変数を設定
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
            return temp_file.name
        except Exception as e:
            os.unlink(temp_file.name)
            raise e
    except Exception as e:
        logger.error(f"Failed to setup Google credentials: {e}")
        raise

# FastAPI initialization with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: setup credentials and create client session
    credentials_path = None
    try:
        credentials_path = setup_google_credentials()
        app.state.client_session = aiohttp.ClientSession()
        app.state.tts_client = texttospeech.TextToSpeechClient()
        app.state.speech_client = speech.SpeechClient()
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
        "http://localhost:3000",  # 開発環境用
        "https://your-song-creator-web.onrender.com"  # 本番環境のフロントエンドURL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # WebSocket用に追加
)

# Interview questions
QUESTIONS = [
    "あなたの青春時代を一言で表すと？",
    "その時期にあなたが最も夢中になっていたものは？",
    "青春時代の挫折や失敗を乗り越えた時の気持ちを一言で？",
    "その頃のあなたにとって最も大切だったものは？",
    "今、あの頃の自分に伝えたい言葉を一つ挙げるとしたら？"
]

# Store answers
answers = {}

class TextToSpeech:
    def __init__(self):
        self.client = app.state.tts_client
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
            raise

# Create TTS instance after the app starts
@app.on_event("startup")
async def startup_event():
    app.state.tts = TextToSpeech()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio_path = temp_audio.name
            content = await file.read()
            temp_audio.write(content)

        with open(temp_audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="ja-JP",
        )

        response = app.state.speech_client.recognize(config=config, audio=audio)
        transcription = response.results[0].alternatives[0].transcript if response.results else ""

        return {"transcription": transcription}

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

async def generate_music(themes: list[str], websocket: WebSocket) -> dict:
    """Generate music using Suno AI while maintaining WebSocket connection"""
    suno_api_url = "https://api.goapi.ai/api/suno/v1/music"
    
    headers = {
        "X-API-Key": os.getenv('SUNO_API_KEY'),
        "Content-Type": "application/json"
    }
    
    themes_text = " ".join(themes)
    description = (
        f"Create a nostalgic J-pop song about {themes_text}. "
    )
    
    if len(description) > 200:
        max_themes_length = 100
        if len(themes_text) > max_themes_length:
            themes_text = themes_text[:max_themes_length] + "..."
            description = f"Create a nostalgic J-pop song about {themes_text}. Female JP vocal."
    
    logger.info(f"Generating music with prompt: {description}")
    
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
    
    try:
        # タスク作成
        async with app.state.client_session.post(suno_api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Suno API Error: Status {response.status}")
                logger.error(f"Response: {error_text}")
                raise ValueError(f"Suno API Error: {error_text}")
            
            task_data = await response.json()
            task_id = task_data['data']['task_id']
            logger.info(f"Task ID: {task_id}")
        
        last_progress = -1
        connection_check_counter = 0
        
        while True:
            try:
                # WebSocket接続の維持確認
                connection_check_counter += 1
                if connection_check_counter >= 10:  # 約1秒ごとにkeep-alive
                    await websocket.send_json({
                        "type": "keep_alive"
                    })
                    connection_check_counter = 0

                # 生成状態の確認
                async with app.state.client_session.get(f"{suno_api_url}/{task_id}", headers=headers) as status_response:
                    status_data = await status_response.json()
                    
                    if 'data' not in status_data:
                        logger.error("Invalid response structure")
                        raise ValueError("Invalid response format")

                    status = status_data['data'].get('status')
                    
                    if status == 'completed':
                        # 完了時の出力確認
                        video_url = None
                        
                        # まずoutput内を確認
                        if 'output' in status_data['data']:
                            video_url = status_data['data']['output'].get('video_url')
                        
                        # 次にclipsを確認
                        if not video_url and 'clips' in status_data['data']:
                            clips = status_data['data']['clips']
                            if clips and isinstance(clips, dict):
                                first_clip = next(iter(clips.values()))
                                video_url = first_clip.get('video_url')
                        
                        # 最後にdata直下を確認
                        if not video_url:
                            video_url = status_data['data'].get('video_url')

                        if video_url:
                            logger.info(f"Music generation completed with URL: {video_url}")
                            return {"status": "success", "video_url": video_url}
                        raise ValueError("No video URL found in completed response")

                    elif status == 'failed':
                        error_message = status_data['data'].get('error', 'Unknown error')
                        logger.error(f"Generation failed: {error_message}")
                        return {"status": "error", "error": error_message}

                    elif status == 'processing':
                        progress = status_data['data'].get('progress', 0)
                        if progress != last_progress:
                            logger.info(f"Progress: {progress}%")
                            await websocket.send_json({
                                "type": "generation_progress",
                                "progress": progress
                            })
                            last_progress = progress

                # 短い間隔でチェック
                await asyncio.sleep(0.1)  # 100ミリ秒ごとにチェック
                
            except (aiohttp.ClientError) as e:
                logger.error(f"Network error: {e}")
                await asyncio.sleep(5)
                continue
            except WebSocketDisconnect:
                logger.error("WebSocket disconnected during generation")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return {"status": "error", "error": str(e)}
            
    except Exception as e:
        logger.error(f"Error in music generation: {str(e)}")
        return {"status": "error", "error": str(e)}

async def estimate_audio_duration(text: str) -> float:
    """
    テキストの長さから音声の再生時間を概算する（日本語）
    """
    # 日本語の場合、1文字あたり約0.2秒として概算
    base_duration = len(text) * 0.2
    # 最低2秒、文末の余白として1秒を追加
    return max(2.0, base_duration) + 1.0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("WebSocket connected")
        
        # インタビューのみ1回実行
        try:
            # 開始メッセージを送信
            greeting = "こんにちは！青春ソングを作るためのインタビューを始めましょう。各質問に一言で答えてください。"
            audio_base64 = await app.state.tts.synthesize_speech(greeting)
            greeting_duration = await estimate_audio_duration(greeting)
            
            await websocket.send_json({
                "type": "speech",
                "text": greeting,
                "audio": audio_base64
            })
            
            await asyncio.sleep(greeting_duration)

            # テーマを収集
            themes = []
            
            # Questions and answers
            for i, question in enumerate(QUESTIONS):
                audio_base64 = await app.state.tts.synthesize_speech(question)
                question_duration = await estimate_audio_duration(question)
                
                await websocket.send_json({
                    "type": "speech",
                    "text": question,
                    "audio": audio_base64
                })
                
                await asyncio.sleep(question_duration)
                response = await websocket.receive_text()
                themes.append(response)
                await asyncio.sleep(0.5)

            # 終了メッセージ
            end_message = "ありがとうございます。ミュージックビデオの生成を開始します。"
            audio_base64 = await app.state.tts.synthesize_speech(end_message)
            await websocket.send_json({
                "type": "speech",
                "text": end_message,
                "audio": audio_base64
            })
            
            await asyncio.sleep(await estimate_audio_duration(end_message))
            
            # 楽曲生成開始通知
            await websocket.send_json({
                "type": "status_update",
                "status": "generating_music"
            })

            # 楽曲生成
            music_result = await generate_music(themes, websocket)
            
            if music_result.get("status") == "error":
                await websocket.send_json({
                    "type": "music_error",
                    "data": music_result["error"]
                })
            elif music_result.get("status") == "success":
                # 生成成功時は完了通知を送信
                await websocket.send_json({
                    "type": "music_complete",
                    "data": {
                        "video_url": music_result["video_url"]
                    }
                })
            
            # 処理完了後、明示的に接続を閉じる
            await websocket.close(code=1000)
            logger.info("Interview and music generation completed successfully")
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during interview")
            raise
        except Exception as e:
            logger.error(f"Error in interview process: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": "処理中にエラーが発生しました"
            })
            raise
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        logger.info("WebSocket connection closed")