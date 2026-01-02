from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import MagazineState
from src.config import config

def run_safety(state: MagazineState) -> dict:
    print("--- [2] Safety Filter: 유해성 검사 중... ---")
    llm = config.get_llm()
    
    # 프롬프트: 넌 보안관이야.
    prompt = ChatPromptTemplate.from_template(
        """
        Check the following text for PII (Personally Identifiable Information), hate speech, sexual content, or dangerous instructions.
        
        Text: {user_input}
        
        If safe, return "SAFE".
        If unsafe, return "UNSAFE" and a brief reason.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"user_input": state["user_input"]}).strip()
    
    is_safe = "SAFE" in result
    print(f"🛡️ 안전성 결과: {result}")

    return {
        "safety_check": "SAFE" if is_safe else "UNSAFE",
        "logs": [f"Safety: {result}"]
    }