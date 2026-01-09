# src/agents/editor.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state import MagazineState
from src.config import config

def run_editor(state: MagazineState) -> dict:
    """
    [Unified Structure Refactor]
    Planner의 전략에 따라 기사 원고를 작성합니다.
    state['articles'][id]['manuscript'] 에 결과를 저장합니다.
    """
    print("--- [4] Editor Agent: English Article Generation (Unified) ---")
    
    articles = state.get("articles", {})
    llm = config.get_llm()
    parser = JsonOutputParser()

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Professional Editor for a High-End English Magazine (like Kinfolk, Vogue, or Time).
        
        {mode_instruction}

        !!! CRITICAL RULES !!!
        1. **ENGLISH OUTPUT ONLY**: The final result must be in **ENGLISH**.
        2. **PRIMARY TASK**: Correct spelling, grammar, punctuation, and spacing errors.
        3. **NO HALLUCINATIONS**: Do NOT invent new fictional stories, entities, places, dates, or numbers. Keep the facts intact.
        4. **NO EMOJIS**: High-end magazines do not use emojis. Even if the source has them, or if you aim for a friendly tone, NEVER use emojis.
        5. **TONE POLICY**: Preserve the user's original voice and style as much as possible.
        6. **EXCEPTION**: ONLY modify the tone if the current text is **critically mismatched** with the [Planner Strategy] (e.g., Slang in a Medical article). Otherwise, keep it as is.

        [Input Data]
        - User Request: {user_request}
        - Planner Strategy: {target_tone}
        - Image Context: {image_desc} (Use for Caption)
        - Layout Type: {layout_type}

        [Directives]
        1. **Tone Reference (Style Guide)**:
            Match the tone to the [Planner Strategy]. Planner uses TYPE codes:
            
            **Primary Types (from Planner)**:
            - **TYPE_FASHION_COVER**: Elegant & Lyrical - Poetic, flowing, sophisticated
            - **TYPE_STREET_VIBE**: Bold & Energetic - Punchy, active voice, strong verbs
            - **TYPE_EDITORIAL_SPLIT**: Analytical & Professional - Precise, objective, logic-focused
            - **TYPE_LUXURY_PRODUCT**: Minimalist & Clean - Concise, dry, direct
            
            **Fallback Styles (if user provides direct preference)**:
            - **Elegant**: Poetic, flowing, sophisticated
            - **Bold**: Punchy, active voice, strong verbs
            - **Analytical**: Precise, objective, logic-focused
            - **Friendly**: Warm, inviting, uses "You"
            - **Witty**: Clever wordplay, sharp humor
            - **Dramatic**: Suspenseful, emotional, sensory
            - **Minimalist**: Concise, dry, direct
            - **Nostalgic**: Evocative, cozy, retro
            
            **IMPORTANT**: Do not force this style if the original text is already acceptable.

        2. **Smart Captioning (The Bridge)**:
            - **Rule**: Do NOT mention the image in the 'Body'.
            - **Task**: Write a separate 'Caption' connecting the [Image Context] with the core theme of the text.
            - **Length**: Max 15 words.
            - **Formula**: "[Visual Detail from Image], [Connection Verb] the article's theme of [Core Topic]."
            - **Example**: "The golden sunset at Uluwatu, reflecting the article's theme of inner peace."

        3. **Adaptive Formatting (Crucial for Layout)**:
            - **Headline**: Max 7 words. Catchy.
            - **Body Structure**: Adapt based on {layout_type}.
                - **If {layout_type} is Long-form (e.g., Feature, Essay)**: Keep the length. Break into readable paragraphs using double line breaks (\\n\\n).
                - **If {layout_type} is Short-form (e.g., Editorial, Brief)**: Concise paragraphs. No fluff. Direct impact.
                - **If {layout_type} is Interview**: Strictly maintain the Question & Answer format.
            - **Output**: JSON format ONLY. Do not include markdown tags.

        [Output JSON format]
        {{
            "headline": "English Title",
            "subhead": "Subtitle",
            "body": "English content...",
            "pull_quote": "Key quote",
            "caption": "Connection between image and text",
            "tags": ["Tag1", "Tag2"]
        }}
        """
    )
    
    chain = prompt | llm | parser

    for a_id, article in articles.items():
        # [Dependency Check] Planner 데이터 존재 여부 확인
        plan = article.get("plan")
        if not plan:
            print(f"⚠️ [Editor] 기사 ID {a_id}: Planner가 실행되지 않아 기본 설정으로 진행합니다.")
            plan = {} # 빈 딕셔너리로 초기화하여 에러 방지

        # 데이터 로드
        req_text = article.get("request", "")
        title_text = article.get("title", "Untitled")
        is_gen = article.get("is_generated", True)
        
        # Planner & Vision 데이터
        target_tone = plan.get("selected_type") or article.get("style", "Elegant")
        vision = article.get("vision_analysis", {})
        image_desc = vision.get("metadata", {}).get("description", "Visual")
        
        # --- [Case 1: 사용자 직접 입력 보존] ---
        if not is_gen:
            print(f"   -> 👤 사용자 본문 유지 (ID: {a_id})")
            article["manuscript"] = {
                "headline": title_text,
                "subhead": "Original Draft",
                "body": req_text,
                "pull_quote": "",
                "caption": f"Visual context for {title_text}",
                "tags": [target_tone]
            }
            continue

        # --- [Case 2: AI 자동 생성] ---
        # 모드 결정 (긴 텍스트: 교정 / 짧은 텍스트: 생성) , 프롬프트에 들여쓰기는 토큰 비효율이래.
        is_polish_mode = len(req_text.strip()) > 50
        
        if is_polish_mode:
            mode_instruction = """MODE: Proofreading & Minor Fixes (User provided a draft)
- Preserve the original meaning and nuances.
- Focus strictly on correcting grammar, spelling, and phrasing.
- Only adjust the tone if it is critically mismatched."""
        else:
            mode_instruction ="""MODE: Creative Writing (User provided keywords)
- Generate a full, captivating magazine article from scratch.
- Expand on ideas to create a rich narrative fitting the target tone."""

        print(f"✍️ Editor 작성 중... ID:{a_id} | 모드:{'Polish' if is_polish_mode else 'Create'}")

        try:
            generated = chain.invoke({
                "mode_instruction": mode_instruction,
                "user_request": req_text,
                "target_tone": target_tone,
                "image_desc": image_desc,
                "layout_type": plan.get("selected_type", "Standard")
            })

            # ID 및 제목 보정
            if title_text and title_text != "Untitled":
                generated["headline"] = title_text
            
            # ✅ 결과 저장
            article["manuscript"] = generated

        except Exception as e:
            print(f"❌ Editor Error (ID: {a_id}): {e}")
            article["manuscript"] = {
                "headline": title_text,
                "subhead": "Error",
                "body": f"generation failed: {req_text}",
                "caption": "Error",
                "tags": ["Error"]
            }

    return {"articles": articles}