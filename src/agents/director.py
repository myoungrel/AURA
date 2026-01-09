# src/agents/director.py
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state import MagazineState
from src.config import config

def run_director(state: MagazineState) -> dict:
    """
    [Unified Structure Refactor]
    레이아웃 및 비주얼 분석 결과를 통합하여 최종 디자인 스펙을 생성합니다.
    state['articles'][id]['design_spec'] 에 결과를 저장합니다.
    """
    print("--- [5] Art Director: Generating SDUI Design Spec (Unified) ---")
    
    articles = state.get("articles", {})
    llm = config.get_llm()
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a World-Class Art Director & UI/UX Designer.
        Create a **JSON Design Specification (SDUI)** based on the input Strategy and Visual Analysis.
        
        [Input Data]
        - **Layout Mode**: {layout_mode}
        - **Design Strategy (Type)**: {target_tone}
        - **Extracted Colors**: {extracted_colors}
        - **Safe Areas**: {safe_areas}
        - **Font Idea**: {font_vibe}

        [Design Rules (Few-Shot)]
        Apply the style strictly based on the Strategy Type:
        - **Elegant**: Serif fonts (Playfair Display, Crimson Text), High contrast, Overlay opacity 0.3
        - **Bold**: Sans-Serif (Oswald, Bebas Neue), Vivid accent colors, Italic headlines
        - **Analytical**: Clean Sans-Serif (Roboto, Inter), Grid layout, High legibility
        - **Friendly**: Rounded Sans (Nunito, Quicksand), Pastel tones, Card layout
        - **Witty**: Retro Serif (Merriweather, Courier Prime), Brutalist layout, Stark borders
        - **Dramatic**: Cinematic Serif (Cinzel, Cormorant), Dark mode, High fade gradients
        - **Minimalist**: Modern Sans (Inter, Work Sans), Huge whitespace, Small typography
        - **Nostalgic**: Retro font (Courier Prime, Special Elite), Sepia/Grainy filters, Polaroid style

        [Critical Instructions]
        1. **Color Selection**: Choose the most vibrant color from [Extracted Colors] as 'primary'. ALL colors MUST be valid HEX codes (e.g., #FF5733).
        2. **Font Selection**: Use actual font names with fallbacks (e.g., "Playfair Display, serif").
        3. **Layout**: Analyze [Safe Areas] for text alignment.
        4. **Variety**: Adapt all styling values to match the strategy - avoid repetitive choices.

        [Output Requirements]
        Return ONLY valid JSON. Replace ALL angle-bracket placeholders with actual computed values.

        {{
            "layout_strategy": "<hero_overlay_smart or separated_grid or magazine_column>",
            "theme": {{
                "mood": "{target_tone}",
                "colors": {{
                    "primary": "<most vibrant HEX from extracted_colors>",
                    "text_main": "<#FFFFFF for dark bg or #000000 for light bg>",
                    "text_sub": "<complementary HEX>"
                }},
                "fonts": {{
                    "title": "<Font Name, fallback>",
                    "body": "<Font Name, fallback>"
                }}
            }},
            "layout_config": {{
                "text_alignment": "<text-left or text-right or text-center>",
                "overlay_opacity": "<bg-black/20 to bg-black/70>"
            }},
            "components_style": {{
                "content_box": {{
                    "bg_color": "<bg-black/XX or bg-white/XX>",
                    "padding": "<p-4 to p-12>",
                    "border_radius": "<rounded-none to rounded-3xl>",
                    "backdrop_blur": "<backdrop-blur-none to backdrop-blur-xl>",
                    "shadow": "<shadow-none to shadow-2xl>"
                }},
                "headline": {{
                    "size": "<text-4xl to text-7xl, use responsive>",
                    "weight": "<font-normal to font-black>"
                }}
            }}
        }}
        """
    )
    
    chain = prompt | llm | parser

    for a_id, article in articles.items():
        # [Strict Dependency Check]
        plan = article.get("plan")
        vision = article.get("vision_analysis")
        
        if not plan or not vision:
            print(f"⚠️ [Director] 기사 ID {a_id}: 필수 데이터(Planner/Vision) 누락.")
            # Fallback spec
            article["design_spec"] = {
                "layout_strategy": "Separated",
                "theme": {"mood": "Fallback"}, 
                "components_style": {}
            }
            continue

        # 데이터 매핑
        target_tone = plan.get("selected_type", "Elegant")
        layout_guide = plan.get("layout_guide", {})
        
        strategy_data = vision.get("layout_strategy", {})
        layout_mode = strategy_data.get("recommendation", "Overlay")
        metadata = vision.get("metadata", {})
        
        extracted_colors = metadata.get("dominant_colors", ["#000000"])
        safe_areas = vision.get("safe_areas", [])
        font_vibe = layout_guide.get("font_theme", "Sans-serif")

        print(f"🎨 디자인 중... ID:{a_id} | 모드:{layout_mode}")

        try:
            spec = chain.invoke({
                "layout_mode": layout_mode,
                "target_tone": target_tone,
                "extracted_colors": str(extracted_colors),
                "safe_areas": str(safe_areas),
                "font_vibe": font_vibe
            })
            
            # ✅ 결과 저장
            article["design_spec"] = spec
            
        except Exception as e:
            print(f"❌ Director Error (ID: {a_id}): {e}")
            article["design_spec"] = {"theme": {"mood": "Error"}}

    return {"articles": articles}