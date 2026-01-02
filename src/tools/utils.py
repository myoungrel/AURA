# src/tools/utils.py
import os
import base64

def load_image_as_base64(image_path: str) -> str:
    """이미지 경로를 받아서 Base64 문자열로 반환"""
    if not image_path or image_path == "no_image":
        return None
        
    if not os.path.exists(image_path):
        print(f"❌ 이미지를 찾을 수 없음: {image_path}")
        return None
        
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ 이미지 인코딩 에러: {e}")
        return None