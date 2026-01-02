import os
import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io
import base64

# ==========================================
# 1. 상태(State) 정의
# ==========================================
class MagazineState(TypedDict):
    content: str
    images: list
    category: str
    layout_plan: str
    final_html: str

# ==========================================
# 2. 헬퍼 함수 (이미지 변환)
# ==========================================
def image_to_base64(image):
    """PIL 이미지를 HTML용 base64 문자열로 변환 (JPEG 압축 적용)"""
    buffered = io.BytesIO()
    # 이미지를 RGB로 변환 (PNG 투명 배경 이슈 방지)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=85) # 용량 최적화
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# ==========================================
# 3. 노드(에이전트) 정의
# ==========================================

# [노드 1] 편집장 (Supervisor): 이미지 개수에 따라 레이아웃 강제 결정
def supervisor_node(state: MagazineState):
    category = state['category']
    img_count = len(state['images'])
    
    # ★ 수정됨: 이미지 개수에 따른 확실한 분기 로직
    if img_count >= 4:
        plan = "Type C (Briefs)" # 이미지가 많으면 무조건 그리드형
        reason = f"이미지가 {img_count}장이므로 브리핑(Briefs) 레이아웃이 적합함."
    elif img_count >= 2:
        plan = "Type B (Split)"
        reason = "이미지가 2~3장이므로 분할(Split) 레이아웃이 적합함."
    else:
        plan = "Type A (Feature)"
        reason = "이미지가 1장이므로 피처(Feature) 레이아웃이 적합함."
        
    print(f"편집장 결정: {plan} (이유: {reason})")
    return {"layout_plan": plan}

# [노드 2] 디자이너 (HTML Coder): 모든 이미지를 강제로 사용하게 함
def designer_node(state: MagazineState):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"final_html": "<div style='color:red'>Error: API Key Missing</div>"}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
    
    # 이미지 플레이스홀더 목록 생성 (예: [%%IMAGE_0%%, %%IMAGE_1%%...])
    img_placeholders = [f"%%IMAGE_{i}%%" for i in range(len(state['images']))]
    placeholders_str = ", ".join(img_placeholders)

    # ★ 수정됨: 강력한 프롬프트 (모든 이미지 사용 강제)
    prompt_text = f"""
    당신은 세계적인 매거진(Time, Vogue)의 수석 웹 개발자입니다.
    
    [작업 정보]
    - 카테고리: {state['category']}
    - 레이아웃 타입: {state['layout_plan']} (이 구조를 엄격히 따를 것)
    - **사용해야 할 이미지 토큰들: {placeholders_str}**
    
    [필수 규칙 - 어길 시 해고]
    1. **모든 이미지 사용:** 위 '이미지 토큰들'을 <img> 태그의 src 속성에 하나도 빠짐없이 다 넣으세요.
    2. **절대 규칙:** src에는 오직 `%%IMAGE_숫자%%` 형식만 들어가야 합니다. (Base64 코드나 실제 URL 넣지 마세요)
    3. **레이아웃:**
       - Type C (Briefs)일 경우: 첫 번째 이미지는 크게 메인으로, 나머지 이미지들은 하단 그리드(Briefs Grid)에 작게 배치하세요.
    4. **스타일:** CSS는 반드시 <style> 태그 안에 포함하세요. Tailwind 쓰지 마세요.
    5. **출력:** 오직 HTML 코드만 출력하세요. (마크다운 ```html 포함 금지)
    
    [내용]
    {state['content']}
    """

    msg_content = [{"type": "text", "text": prompt_text}]
    
    # Gemini 호출
    response = llm.invoke([HumanMessage(content=msg_content)])
    
    html = response.content
    # 마크다운 방어 코드
    html = html.replace("```html", "").replace("```", "").strip()
    
    # ★ 수정됨: 이미지 치환 로직 (깨짐 방지)
    for i, img in enumerate(state['images']):
        placeholder = f"%%IMAGE_{i}%%"
        
        # 1. Base64 변환
        base64_data = image_to_base64(img)
        
        # 2. HTML 내 치환 (혹시 모를 공백 제거)
        if placeholder in html:
            html = html.replace(placeholder, base64_data)
        else:
            # AI가 실수를 했을 경우를 대비한 비상 대책 (강제 삽입)
            print(f"경고: AI가 {placeholder}를 누락했습니다. 하단에 강제 추가합니다.")
            html = html.replace("</body>", f"<div style='margin:20px'><img src='{base64_data}' width='200'></div></body>")
            
    return {"final_html": html}

# ==========================================
# 4. 그래프 조립
# ==========================================
def build_graph():
    workflow = StateGraph(MagazineState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("designer", designer_node)
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "designer")
    workflow.add_edge("designer", END)
    return workflow.compile()

# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(layout="wide", page_title="LangGraph Magazine")

# CSS 주입 (미리보기 화면 스타일 잡기)
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    h1 { color: #d32f2f; font-family: 'serif'; }
</style>
""", unsafe_allow_html=True)

st.title("LangGraph AI Magazine Editor 📰")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    category = st.selectbox("Category", ["SCIENCE", "BUSINESS", "ARTS"])
    st.info("이미지를 4장 이상 업로드하면 자동으로 'Briefs(단신)' 레이아웃으로 변경됩니다.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Data")
    text = st.text_area("Content", height=300, placeholder="기사 내용을 입력하세요...")
    files = st.file_uploader("Images (4장 권장)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if st.button("Generate Layout", type="primary", use_container_width=True):
        if not api_key or not text:
            st.error("API Key와 텍스트를 입력해주세요.")
        else:
            imgs = [Image.open(f) for f in files] if files else []
            app = build_graph()
            inputs = {"content": text, "images": imgs, "category": category}
            
            with st.spinner("AI 편집국이 일하는 중... (편집장 -> 디자이너)"):
                result = app.invoke(inputs)
                st.session_state['html'] = result['final_html']
                st.session_state['layout_plan'] = result['layout_plan']
                st.success(f"생성 완료! 적용된 레이아웃: {result['layout_plan']}")

with col2:
    st.subheader("2. Result Preview")
    if 'html' in st.session_state:
        # 다운로드 버튼
        st.download_button(
            label="Download HTML",
            data=st.session_state['html'],
            file_name="magazine.html",
            mime="text/html"
        )
        # HTML 렌더링 (스크롤 가능하게 높이 지정)
        st.components.v1.html(st.session_state['html'], height=800, scrolling=True)
    else:
        st.info("왼쪽에서 데이터를 입력하고 버튼을 눌러주세요.")