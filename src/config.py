import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# .env 파일 로드
load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Gemini 1.5 Pro 모델 설정 (가장 똑똑한 녀석!)
    MODEL_NAME = "gemini-2.0-flash"

    @staticmethod
    def get_llm():
        # API 키 체크
        if not Config.GOOGLE_API_KEY:
            print("⚠️ [경고] .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")
            
        return ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.7
        )

# ---------------------------------------------------------
# 👇 [중요] 이 부분이 빠졌었어! 클래스를 실체화(인스턴스)해서 내보내야 해.
# ---------------------------------------------------------
config = Config()