from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state import MagazineState
from src.config import config

def run_director(state: MagazineState) -> dict:
    print("--- [5] Art Director: 디자인 전략 수립 중... ---")
    llm = config.get_llm()
    
    # JSON 파서 설정 (LLM이 JSON 형식을 잘 지키도록 유도)
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Art Director for a high-end magazine.
        Based on the article and image analysis, define the design strategy.
        
        [Article]: {manuscript}
        [Image Mood]: {vision_result}
        
        Return a JSON object with the following keys:
        - "layout_style": (e.g., "Minimalist", "Editorial Grid", "Hero Header")
        - "color_palette": (List of 3 Hex codes e.g., ["#FFFFFF", "#000000", "#FF5733"])
        - "font_pairing": (e.g., "Serif for Headings, Sans-serif for Body")
        - "vibe_description": (Short instruction for the web developer)
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    
    try:
        design_plan = chain.invoke({
            "manuscript": state["manuscript"],
            "vision_result": state.get("vision_result", ""),
            "format_instructions": parser.get_format_instructions()
        })
        print(f"🎨 디자인 전략: {design_plan.get('layout_style')}")
    except Exception as e:
        print(f"Director Error: {e}")
        design_plan = {"layout_style": "Simple", "color_palette": ["#000000"], "vibe_description": "Clean"}

    return {
        "design_plan": design_plan,
        "logs": ["Director: 디자인 가이드라인(JSON) 수립 완료"]
    }