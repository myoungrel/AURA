from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import MagazineState
from src.config import config

def run_router(state: MagazineState) -> dict:
    print("--- [1] Intent Router: 의도 파악 중... ---")
    llm = config.get_llm()
    
    # 프롬프트: 넌 분류기야.
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Intent Classifier.
        Determine if the user's input is related to creating a magazine, article, blog post, or design content.
        
        User Input: {user_input}
        
        Return ONLY one word: "create_magazine" if relevant, or "others" if not.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"user_input": state["user_input"]}).strip()
    
    print(f"🧐 파악된 의도: {intent}")
    
    return {
        "intent": intent,
        "logs": [f"Router: 의도 파악 완료 ({intent})"]
    }