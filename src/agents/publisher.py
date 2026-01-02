from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import MagazineState
from src.config import config

def run_publisher(state: MagazineState) -> dict:
    print("--- [6] Publisher: HTML/CSS 코딩 중... ---")
    llm = config.get_llm()
    
    # 디자인 계획을 문자열로 변환
    design_plan = str(state.get("design_plan", {}))
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a World-Class Frontend Developer.
        Create a single-file HTML using Tailwind CSS based on the design plan and manuscript.
        
        [Design Plan]: {design_plan}
        [Manuscript]: {manuscript}
        
        CRITICAL INSTRUCTIONS:
        1. Use Tailwind CSS via CDN (<script src="https://cdn.tailwindcss.com"></script>).
        2. Make it responsive and beautiful (Magazine quality).
        3. Use the 'manuscript' content fully.
        4. **IMPORTANT**: For the main image, use exactly this source: src="{{IMAGE_PLACEHOLDER}}"
           (Do NOT put real base64 code here, just the placeholder text).
        5. Return ONLY the raw HTML code. Do not use Markdown backticks (```html).
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    html_code = chain.invoke({
        "design_plan": design_plan,
        "manuscript": state["manuscript"]
    })
    
    # 🧹 마크다운 백틱 제거 (가끔 LLM이 습관적으로 붙임)
    html_code = html_code.replace("```html", "").replace("```", "").strip()
    
    # 💉 [핵심 기술] 메모리에 있는 Base64 이미지를 HTML에 주입!
    if state.get("image_data"):
        image_src = f"data:image/jpeg;base64,{state['image_data']}"
        html_code = html_code.replace("{{IMAGE_PLACEHOLDER}}", image_src)
        # 혹시 모를 URL 인코딩 문자 처리
        html_code = html_code.replace("%7B%7BIMAGE_PLACEHOLDER%7D%7D", image_src)
    
    print("💻 HTML 생성 완료 (이미지 주입됨)")

    return {
        "html_code": html_code,
        "logs": ["Publisher: HTML/CSS 코딩 및 이미지 병합 완료"]
    }