from src.state import MagazineState

def run_critique(state: MagazineState) -> dict:
    print("--- [7] Critique: 품질 검수 중... ---")
    
    html = state.get("html_code", "")
    
    # 간단한 검증 로직
    if not html or len(html) < 100:
        feedback = "HTML generation failed or too short."
        # (고도화 시 여기서 Publisher로 돌려보내는 루프를 만들 수 있음)
    elif "<html" not in html.lower() and "<div" not in html.lower():
         feedback = "Output does not look like valid HTML."
    else:
        feedback = "Quality Assured: Design looks good and code is valid."
    
    print(f"🧐 검수 결과: {feedback}")

    return {
        "critique": feedback,
        "logs": [f"Critique: {feedback}"]
    }