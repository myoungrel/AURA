from src.state import MagazineState

def run_formatter(state: MagazineState) -> dict:
    print("--- [8] UX Formatter: 최종 결과물 패키징... ---")
    
    # 최종 HTML 가져오기
    final_html = state.get("html_code", "<h1>Error Generating Page</h1>")
    
    # Streamlit에서 렌더링할 때 필요한 메타데이터가 있다면 여기서 추가 가능
    
    return {
        "final_output": final_html,
        "logs": ["Formatter: 최종 렌더링 준비 완료"]
    }