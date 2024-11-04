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
import base64
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

def setup_google_credentials():
    """
    Base64でエンコードされたGoogle Cloud認証情報を
    一時的なJSONファイルとして保存し、環境変数を設定する
    """
    try:
        # Base64エンコードされた認証情報を取得
        encoded_credentials = os.getenv('GOOGLE_CREDENTIALS_BASE64')
        if not encoded_credentials:
            raise ValueError("GOOGLE_CREDENTIALS_BASE64 environment variable is not set")

        # Base64デコード
        credentials_json = base64.b64decode(encoded_credentials).decode('utf-8')
        
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            # JSONを書き込み
            json.dump(json.loads(credentials_json), temp_file, ensure_ascii=False, indent=2)
            temp_file_path = temp_file.name

        # 環境変数を設定
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
        return temp_file_path

    except Exception as e:
        logger.error(f"Error setting up Google credentials: {str(e)}")
        raise

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create aiohttp client session and setup credentials
    credentials_path = None
    try:
        credentials_path = setup_google_credentials()
        app.state.client_session = aiohttp.ClientSession()
        logger.info("Created aiohttp ClientSession and setup Google credentials")
        
        yield
        
    finally:
        # Shutdown: cleanup
        if app.state.client_session:
            await app.state.client_session.close()
        # 一時ファイルの削除
        if credentials_path and os.path.exists(credentials_path):
            try:
                os.unlink(credentials_path)
            except Exception as e:
                logger.error(f"Failed to delete credentials file: {e}")
        logger.info("Cleanup completed")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS - 本番環境用に更新
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-song-creator-frontend.onrender.com",  # 本番環境のフロントエンドURL
        "http://localhost:3000"  # 開発環境用
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 以下、既存のコードをそのまま維持
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
        self.client = texttospeech.TextToSpeechClient()
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

# Create TTS instance
tts = TextToSpeech()

class ThemesRequest(BaseModel):
    themes: List[str]

# ルートエンドポイントを追加（ヘルスチェック用）
@app.get("/api")
async def root():
    return {"status": "ok"}

# 質問エンドポイントの修正
@app.post("/api/questions/{index}")
async def get_question(index: int):
    try:
        if index < 0 or index >= len(QUESTIONS):
            raise HTTPException(status_code=404, detail="Question not found")
        
        question = QUESTIONS[index]
        audio = await tts.synthesize_speech(question)
        
        return JSONResponse(content={
            "question": question,
            "audio": audio,
            "isLast": index == len(QUESTIONS) - 1
        })
    except Exception as e:
        logger.error(f"Error getting question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def check_url_validity(url: str, max_attempts: int = 30) -> bool:
    """URLが有効になるまで待機する"""
    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code == 200:
                return True
        except Exception as e:
            logger.debug(f"URL not yet available ({attempt+1}/{max_attempts}): {str(e)}")
        
        await asyncio.sleep(2)
        attempt += 1
    return False

def get_video_url_from_response(data: dict) -> str | None:
    """レスポンスデータからビデオURLを抽出する"""
    video_url = None
    
    # まずoutput内を確認
    if 'output' in data:
        video_url = data['output'].get('video_url')
    
    # 次にclipsを確認
    if not video_url and 'clips' in data:
        clips = data.get('clips', {})
        if clips and isinstance(clips, dict):
            first_clip = next(iter(clips.values()))
            video_url = first_clip.get('video_url')
    
    # 最後にdata直下を確認
    if not video_url:
        video_url = data.get('video_url')
    
    return video_url

@app.post("/generate-music")
async def generate_music(request: ThemesRequest):
    try:
        suno_api_url = "https://api.goapi.ai/api/suno/v1/music"
        
        headers = {
            "X-API-Key": os.getenv('SUNO_API_KEY'),
            "Content-Type": "application/json"
        }
        
        themes_text = " ".join(request.themes)
        description = f"Create a nostalgic J-pop song about {themes_text}. Female JP vocal."
        
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
        
        # タスクの作成
        response = requests.post(suno_api_url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(f"Suno API Error: Status {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Music generation failed: {response.text}"
            )
        
        task_data = response.json()
        if 'data' not in task_data or 'task_id' not in task_data['data']:
            logger.error(f"Invalid response structure: {task_data}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response from Suno API"
            )
        
        task_id = task_data['data']['task_id']
        logger.info(f"Task ID: {task_id}")
        
        # タスクの完了を待つ
        max_attempts = 120  # 4分まで待機（2秒 × 120回）
        attempt = 0
        last_progress = -1
        
        while attempt < max_attempts:
            status_response = requests.get(
                f"{suno_api_url}/{task_id}",
                headers=headers
            )
            
            if status_response.status_code != 200:
                logger.error(f"Status check failed: {status_response.text}")
                attempt += 1
                await asyncio.sleep(2)
                continue
            
            status_data = status_response.json()
            if 'data' not in status_data:
                logger.error(f"Invalid status response: {status_data}")
                attempt += 1
                await asyncio.sleep(2)
                continue
            
            status = status_data['data'].get('status')
            current_progress = status_data['data'].get('progress', 0)
            
            # 進捗が更新された場合のみログ出力
            if current_progress != last_progress:
                logger.info(f"Generation progress: {current_progress}%")
                last_progress = current_progress
            
            if status == 'completed':
                video_url = get_video_url_from_response(status_data['data'])
                
                if video_url:
                    logger.info(f"Checking URL validity: {video_url}")
                    is_valid = await check_url_validity(video_url)
                    if is_valid:
                        logger.info(f"URL is now valid and accessible")
                        return {
                            "video_url": video_url,
                            "progress": 100,
                            "status": "ready"
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail="Generated URL is not yet accessible"
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="No video URL found in completed response"
                    )
            
            elif status == 'failed':
                error_message = status_data['data'].get('error', 'Unknown error')
                logger.error(f"Generation failed: {error_message}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Music generation failed: {error_message}"
                )
            
            elif status == 'processing':
                progress = status_data['data'].get('progress', 0)
                if attempt % 5 == 0:  # 10秒ごとにログ出力
                    logger.info(f"Still processing... Progress: {progress}%")
            
            attempt += 1
            await asyncio.sleep(2)
        
        raise HTTPException(
            status_code=500,
            detail=f"Generation still in progress after {max_attempts * 2} seconds. Current progress: {current_progress}%"
        )

    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """音声をテキストに変換するエンドポイント"""
    temp_audio_path = None

    try:
        client = speech.SpeechClient()
        
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

        response = client.recognize(config=config, audio=audio)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)