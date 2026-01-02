# src/state.py
from typing import TypedDict, List, Annotated, Optional
import operator

class MagazineState(TypedDict):
    user_input: str
    
    # 👇 [수정] 이제 '경로' 대신 '데이터'를 직접 담을 거야!
    image_data: Optional[str]  # Base64 인코딩된 이미지 문자열
    
    intent: str
    safety_check: str
    vision_result: str
    manuscript: str
    design_plan: str
    html_code: str
    critique: str
    final_output: str
    
    logs: Annotated[List[str], operator.add]