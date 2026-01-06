from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import MagazineState
from src.config import config
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import re

# [추가] 출력 구조 정의
class SafetyCheck(BaseModel):
    is_safe: bool = Field(description="유해성 여부 (True: 안전, False: 위험)")
    reason: str = Field(description="위험 판단 이유 (안전할 경우 'None')")
    pii_detected: list = Field(description="검출된 개인정보 항목들")

def run_safety(state: MagazineState) -> dict:
    print("--- [2] Safety Filter: 유해성 검사 중... ---")
    llm = config.get_llm()

    # 1. Pydantic Parser 설정: LLM이 JSON 형식을 지키도록 강제합니다.
    parser = PydanticOutputParser(pydantic_object=SafetyCheck)
    
    user_input = state.get("user_input", "") 

    # 2. 정규표현식을 이용한 사전 PII 검사 (Email, Phone 등)
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    found_emails = re.findall(email_pattern, user_input)

    # 3. 프롬프트 수정 
    # - {format_instructions}를 추가하여 LLM에게 정확한 JSON 구조를 전달
    # - 단순 "SAFE" 반환이 아닌, 상세한 분석을 요구하도록 페르소나 강화
    prompt = ChatPromptTemplate.from_template(
        """
        You are a strict Security Officer for a publishing company. 
        Analyze the text for PII (names, addresses, IDs), hate speech, Sexual content, Dangerous activities, or inappropriate content.
        
        Text to analyze: {user_input}
        
        {format_instructions}
        """
    ).partial(format_instructions=parser.get_format_instructions()) # Parser가 생성한 지침 삽입
    
    # 4. 체인 구성 및 호출
    # 변경 사항: StrOutputParser() 대신 위에서 정의한 parser를 사용합니다.
    chain = prompt | llm | parser

    try:
        # result는 이제 SafetyCheck 클래스의 인스턴스(객체)가 됩니다.
        result = chain.invoke({"user_input": user_input})
        
        # 5. 정규표현식 결과와 LLM 결과 병합
        # 변경 사항: LLM이 놓칠 수 있는 정규식 패턴(이메일 등)을 최종 결과에 강제로 추가합니다.
        if found_emails:
            result.is_safe = False
            result.pii_detected = list(set(result.pii_detected + found_emails))
            result.reason += " [System] Email pattern detected via Regex."

    except Exception as e:
        # 🚨 [폴백] LLM 호출 실패 시 가장 보수적인(안전한) 판단을 내림
        print(f"❌ Safety Filter Error: {e}")
        result = SafetyCheck(
            is_safe=False, 
            reason="Safety check failed due to system error. (Fallback activated)",
            pii_detected=[]
        )

    print(f"🛡️ 안전성 결과: {'SAFE' if result.is_safe else 'UNSAFE'} (사유: {result.reason})")

    # 6. 최종 State 반환
    # 변경 사항: A가 정의한 state 구조에 맞춰 'safety_check'와 상세 'safety_detail'을 함께 넘깁니다.
    return {
        "safety_check": "SAFE" if result.is_safe else "UNSAFE",
        "safety_detail": result.model_dump(), # 상세 데이터(사유, PII 목록) 저장 (Pydantic V2부터는 .dict() 대신 .model_dump()를 사용. dict 빗금발생)
        "logs": [f"Safety: {result.is_safe}, Reason: {result.reason}"]
    }