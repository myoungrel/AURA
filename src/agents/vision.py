# src/agents/vision.py
from langchain_core.messages import HumanMessage
from src.state import MagazineState
from src.config import config

# 💡 encode_image 함수는 이제 필요 없어! 삭제!

def run_vision(state: MagazineState) -> dict:
    print("--- [3] Vision Agent: 이미지 분석 중... ---")
    
    # State에서 바로 데이터 꺼내기
    base64_image = state.get("image_data")
    
    if not base64_image:
        return {
            "vision_result": "이미지가 없습니다. 텍스트로만 진행합니다.",
            "logs": ["Vision: 이미지 없음 (Skip)"]
        }

    llm = config.get_llm()

    # 프롬프트 구성 (이미지 데이터를 바로 꽂아넣음)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
                Analyze this image for a magazine layout. 
                1. Describe the main subject, lighting, and composition.
                2. Determine the Mood (e.g., Minimalist, Retro, Luxury).
                3. Extract 3 Hex Color Codes from the image.
                """
            },
            {
                "type": "image_url",
                # 👇 이미 인코딩된 문자열을 그대로 사용!
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )
    
    try:
        response = llm.invoke([message])
        print(f"👁️ 비전 분석 완료: {response.content[:50]}...")
        
        return {
            "vision_result": response.content,
            "logs": ["Vision: 메모리 상의 이미지 분석 완료"]
        }
    except Exception as e:
        return {
            "vision_result": f"Error: {str(e)}",
            "logs": [f"Vision Error: {str(e)}"]
        }