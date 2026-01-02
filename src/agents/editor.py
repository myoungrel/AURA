from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import MagazineState
from src.config import config

def run_editor(state: MagazineState) -> dict:
    print("--- [4] Editor Agent: 원고 작성 중... ---")
    llm = config.get_llm()
    
    # 프롬프트: 넌 베테랑 매거진 에디터야.
    prompt = ChatPromptTemplate.from_template(
        """
        You are a professional Magazine Editor. 
        Write a high-quality article based on the User Input and Image Analysis.
        
        [User Input]: {user_input}
        [Image Analysis]: {vision_result}
        
        Instructions:
        1. Create a catchy 'Title' and 'Subtitle'.
        2. Write engaging body content (approx. 3 paragraphs).
        3. Create a 'Key Highlights' section with bullet points.
        4. The Tone should be professional yet engaging (like Vogue or GQ).
        5. Output Format: Markdown.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    manuscript = chain.invoke({
        "user_input": state["user_input"],
        "vision_result": state.get("vision_result", "No image analysis")
    })
    
    print(f"✍️ 작성된 원고: {manuscript[:100]}...")

    return {
        "manuscript": manuscript,
        "logs": ["Editor: 고품질 원고 작성 완료"]
    }