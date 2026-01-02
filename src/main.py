# src/main.py
from langgraph.graph import StateGraph, START, END
from src.state import MagazineState

# 👇 [수정됨] 이제 진짜 에이전트 파일들을 불러옵니다!
from src.agents.router import run_router
from src.agents.safety import run_safety
from src.agents.vision import run_vision
from src.agents.editor import run_editor

# 아직 안 만든 Phase 2 친구들은 일단 더미로 유지 (다음 단계에서 수정 예정)
from src.agents.director import run_director
from src.agents.publisher import run_publisher
from src.agents.critique import run_critique
from src.agents.formatter import run_formatter

def build_graph():
    workflow = StateGraph(MagazineState)

    # 1. 노드 등록
    workflow.add_node("router", run_router)
    workflow.add_node("safety", run_safety)
    workflow.add_node("vision", run_vision)
    workflow.add_node("editor", run_editor)
    
    workflow.add_node("director", run_director)
    workflow.add_node("publisher", run_publisher)
    workflow.add_node("critique", run_critique)
    workflow.add_node("formatter", run_formatter)

    # 2. 엣지 연결 (조건부 분기 추가)
    workflow.add_edge(START, "router")

    # 라우터에서 안전 검사로
    workflow.add_edge("router", "safety")

    # 안전 검사 결과에 따른 분기 (조건부 엣지)
    def check_safety(state):
        return "vision" if state.get("safety_check") == "SAFE" else END
    
    workflow.add_conditional_edges(
        "safety",
        check_safety,
        {"vision": "vision", END: END}
    )

    workflow.add_edge("vision", "editor")
    workflow.add_edge("editor", "director")
    workflow.add_edge("director", "publisher")
    workflow.add_edge("publisher", "critique")
    workflow.add_edge("critique", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()

app_graph = build_graph()