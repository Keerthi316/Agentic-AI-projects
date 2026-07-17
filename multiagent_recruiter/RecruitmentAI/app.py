"""
AI Recruitment Multi-Agent System
Main Streamlit entry point.

Features:
- Dark / Light theme toggle (persisted in session_state)
- Glassmorphism sidebar
- Navigation to all pages
- DB initialization on first run
"""
import sys
import os

# ── make root importable ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from database.db import init_db

# ── page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="AI Recruitment System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── init DB ───────────────────────────────────────────────────────────────
init_db()

# ── session defaults ─────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ── theme helpers ─────────────────────────────────────────────────────────
def _theme_css(dark: bool) -> str:
    if dark:
        return """
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: rgba(255,255,255,0.05);
            --glass-border: rgba(255,255,255,0.1);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-glow: rgba(99,102,241,0.3);
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #3b82f6;
            --sidebar-bg: rgba(15,15,26,0.95);
            --gradient-1: #6366f1;
            --gradient-2: #8b5cf6;
        }
        """
    else:
        return """
        :root {
            --bg-primary: #f8fafc;
            --bg-secondary: #f1f5f9;
            --bg-card: rgba(255,255,255,0.85);
            --glass-border: rgba(0,0,0,0.08);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --accent: #4f46e5;
            --accent-glow: rgba(79,70,229,0.15);
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --info: #2563eb;
            --sidebar-bg: rgba(248,250,252,0.97);
            --gradient-1: #4f46e5;
            --gradient-2: #7c3aed;
        }
        """


def inject_css(dark: bool) -> None:
    theme = _theme_css(dark)
    st.markdown(
        f"""
        <style>
        {theme}

        /* ── global reset ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        /* ── sidebar ── */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text-primary) !important;
        }}

        /* ── glassmorphism cards ── */
        .glass-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.12);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        }}

        /* ── metric cards ── */
        .metric-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-2px); }}
        .metric-value {{
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--gradient-1), var(--gradient-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metric-label {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.3rem;
        }}

        /* ── gradient header ── */
        .gradient-header {{
            background: linear-gradient(135deg, var(--gradient-1) 0%, var(--gradient-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        /* ── workflow node ── */
        .workflow-node {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 600;
            border: 2px solid var(--glass-border);
            background: var(--bg-card);
            margin: 0.25rem 0;
            transition: all 0.3s;
        }}
        .workflow-node.active {{
            border-color: var(--accent);
            box-shadow: 0 0 16px var(--accent-glow);
            background: var(--accent-glow);
            animation: glow-pulse 1.5s ease-in-out infinite;
        }}
        .workflow-node.done {{
            border-color: var(--success);
            background: rgba(34,197,94,0.1);
        }}

        @keyframes glow-pulse {{
            0%, 100% {{ box-shadow: 0 0 8px var(--accent-glow); }}
            50% {{ box-shadow: 0 0 24px var(--accent-glow); }}
        }}

        /* ── buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, var(--gradient-1), var(--gradient-2)) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.4rem !important;
            font-weight: 600 !important;
            transition: opacity 0.2s, transform 0.1s !important;
        }}
        .stButton > button:hover {{
            opacity: 0.9 !important;
            transform: translateY(-1px) !important;
        }}

        /* ── log panel ── */
        .log-panel {{
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8rem;
            max-height: 300px;
            overflow-y: auto;
            color: var(--text-secondary);
        }}
        .log-entry {{ margin-bottom: 0.3rem; }}
        .log-info {{ color: #60a5fa; }}
        .log-warning {{ color: #fbbf24; }}
        .log-error {{ color: #f87171; }}

        /* ── score badge ── */
        .score-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
        }}

        /* ── nav button in sidebar ── */
        .nav-btn {{
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: none;
            background: transparent;
            color: var(--text-primary);
            font-size: 0.95rem;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .nav-btn:hover, .nav-btn.active {{
            background: var(--accent-glow);
        }}

        /* ── hide default Streamlit elements ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        [data-testid="stDecoration"] {{ display: none; }}

        /* ── input fields ── */
        .stTextArea textarea, .stTextInput input {{
            background: var(--bg-card) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }}

        /* ── expander ── */
        [data-testid="stExpander"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 10px !important;
        }}

        /* ── scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--glass-border); border-radius: 3px; }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ── sidebar ───────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    inject_css(st.session_state.dark_mode)

    with st.sidebar:
        # Logo / title
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem;">
                <div style="font-size:2.5rem;">🤖</div>
                <div class="gradient-header" style="font-size:1.1rem; margin-top:0.3rem;">
                    RecruitmentAI
                </div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.2rem;">
                    Multi-Agent System
                </div>
            </div>
            <hr style="border-color:var(--glass-border); margin:0.75rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # Theme toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"<span style='font-size:0.85rem; color:var(--text-secondary);'>{'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'} Theme</span>",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("⇄", key="theme_toggle", help="Toggle theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

        st.markdown("<hr style='border-color:var(--glass-border);'>", unsafe_allow_html=True)

        # Navigation
        nav_items = [
            ("📊", "Dashboard"),
            ("🎯", "Recruitment"),
            ("📈", "Evaluation"),
            ("📜", "History"),
        ]
        page = st.session_state.page
        for icon, name in nav_items:
            is_active = page == name
            style = (
                "background:var(--accent-glow); border-left:3px solid var(--accent);"
                if is_active
                else "border-left:3px solid transparent;"
            )
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
            ):
                st.session_state.page = name
                st.rerun()

        st.markdown("<hr style='border-color:var(--glass-border);'>", unsafe_allow_html=True)

        # Status / version
        st.markdown(
            """
            <div style="font-size:0.75rem; color:var(--text-secondary); text-align:center; padding:0.5rem 0;">
                v1.0.0 &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; OpenRouter
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.page


# ── page router ───────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if page == "Dashboard":
        from pages.Dashboard import show
        show()
    elif page == "Recruitment":
        from pages.Recruitment import show
        show()
    elif page == "Evaluation":
        from pages.Evaluation import show
        show()
    elif page == "History":
        from pages.History import show
        show()


if __name__ == "__main__":
    main()
