"""
Streamlit Frontend for AI-Powered Quiz Generator
Provides an interactive UI for uploading PPT files and taking AI-generated quizzes
"""

import streamlit as st
import os
import sys
import tempfile
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ppt_parser import extract_text_from_pptx, get_combined_text
from quiz_generator import generate_with_fallback

# Page configuration
st.set_page_config(
    page_title="QuizForge - AI Quiz Generator",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container */
    .main > div {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #1e293b;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    
    /* Cards */
    .stCard {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.2s;
    }
    .upload-area:hover {
        border-color: #6366f1;
        background: #eef2ff;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #6366f1 !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Radio options */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .stRadio > div > label {
        padding: 12px 16px;
        border: 1.5px solid #e2e8f0;
        border-radius: 8px;
        background: white;
        transition: all 0.2s;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        border-color: #6366f1;
        background: #eef2ff;
    }
    .stRadio > div > label[data-selected="true"] {
        border-color: #6366f1;
        background: #eef2ff;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stMetric label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    .stMetric .metric-value {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eef2ff !important;
        color: #6366f1 !important;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #e2e8f0;
    }
    
    /* Success/Error boxes */
    .custom-success {
        background: #d1fae5;
        border: 1px solid #a7f3d0;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
    }
    .custom-error {
        background: #fee2e2;
        border: 1px solid #fecaca;
        padding: 1rem;
        border-radius: 8px;
        color: #991b1b;
    }
    .custom-info {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        padding: 1rem;
        border-radius: 8px;
        color: #4338ca;
    }
    
    /* Question card */
    .question-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .question-number {
        font-size: 0.8rem;
        font-weight: 600;
        color: #1e293b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .question-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0.5rem 0 1rem;
        line-height: 1.6;
    }
    .difficulty-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-simple {
        background: #d1fae5;
        color: #065f46;
    }
    .badge-medium {
        background: #fef3c7;
        color: #92400e;
    }
    .badge-complex {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Explanation box */
    .explanation-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    .explanation-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .explanation-text {
        color: #0f172a;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    .distractor-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.85rem;
        color: #1e293b;
    }
    .distractor-item:last-child {
        border-bottom: none;
    }
    .distractor-label {
        font-weight: 700;
        color: #dc2626;
    }
    
    /* Score circle */
    .score-circle-container {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
    }
    .score-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    .score-circle-bg {
        position: absolute;
        inset: 0;
        border-radius: 50%;
    }
    .score-circle-inner {
        position: relative;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 1;
    }
    .score-big {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
    }
    .score-total {
        font-size: 1rem;
        color: #1e293b;
    }
    
    /* Result item */
    .result-item {
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    .result-item-correct {
        border-left: 4px solid #10b981;
    }
    .result-item-wrong {
        border-left: 4px solid #ef4444;
    }
    
    /* Result text contrast fixes */
    .result-item p {
        color: #0f172a !important;
    }
    
    /* Responsive */
    @media (max-width: 640px) {
        .app-header h1 { font-size: 1.5rem; }
        .question-text { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = 'upload'  # upload, config, quiz, results
if 'file_id' not in st.session_state:
    st.session_state.file_id = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'slide_count' not in st.session_state:
    st.session_state.slide_count = 0
if 'slides_preview' not in st.session_state:
    st.session_state.slides_preview = []
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = False
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 'medium'
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 10


# ===== HEADER =====
st.markdown("""
<div class="app-header">
    <h1>🎯 QuizForge</h1>
    <p>AI-Powered Quiz Generator — Turn your presentations into interactive quizzes</p>
</div>
""", unsafe_allow_html=True)


# ===== UPLOAD SECTION =====
def render_upload_section():
    st.markdown("### 📁 Upload Your Presentation")
    st.markdown("Drop a PowerPoint file to generate AI-powered quizzes")
    
    uploaded_file = st.file_uploader(
        "Choose a .pptx file",
        type=['pptx'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner("📖 Analyzing presentation..."):
            try:
                result = extract_text_from_pptx(tmp_path)
                st.session_state.file_id = tmp_path
                st.session_state.file_name = uploaded_file.name
                st.session_state.slide_count = result['slide_count']
                st.session_state.slides_preview = result['slides'][:3]
                
                st.markdown(f"""
                <div class="custom-success">
                    ✅ <strong>{uploaded_file.name}</strong> uploaded successfully — {result['slide_count']} slides detected
                </div>
                """, unsafe_allow_html=True)
                
                # Show preview
                with st.expander("📄 Content Preview", expanded=True):
                    st.markdown(f"**Slides:** {result['slide_count']}")
                    for slide in result['slides'][:3]:
                        preview_text = slide['text'][:200]
                        if len(slide['text']) > 200:
                            preview_text += '...'
                        if slide['text'] != '(No text content)':
                            st.markdown(f"""
                            <div style="padding: 8px 12px; background: #f8fafc; border-radius: 8px; margin-bottom: 8px; border: 1px solid #e2e8f0;">
                                <span style="font-weight:700; color:#6366f1;">Slide {slide['slide_number']}</span>
                                <p style="margin:4px 0 0; color:#334155; font-size:0.85rem;">{preview_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"*Slide {slide['slide_number']}: (No text content)*")
                    
                    if result['slide_count'] > 3:
                        st.info(f"... and {result['slide_count'] - 3} more slides")
                
                # Proceed button
                if st.button("🚀 Continue to Quiz Configuration", use_container_width=True, type="primary"):
                    st.session_state.quiz_state = 'config'
                    st.rerun()
                    
            except Exception as e:
                st.markdown(f"""
                <div class="custom-error">❌ Error: {str(e)}</div>
                """, unsafe_allow_html=True)
                os.unlink(tmp_path)


# ===== CONFIG SECTION =====
def render_config_section():
    st.markdown("### ⚙️ Configure Your Quiz")
    st.markdown("Customize the quiz parameters to match your needs")
    
    # Show file info
    st.markdown(f"""
    <div class="custom-info">
        📁 <strong>{st.session_state.file_name}</strong> — {st.session_state.slide_count} slides
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Number of Questions**")
        num_q = st.slider(
            "Questions",
            min_value=5,
            max_value=30,
            value=st.session_state.num_questions,
            step=1,
            label_visibility="collapsed"
        )
        st.session_state.num_questions = num_q
        st.caption(f"Choose between 5 and 30 questions (selected: {num_q})")
    
    with col2:
        st.markdown("**Difficulty Level**")
        difficulty = st.selectbox(
            "Difficulty",
            options=["Simple", "Medium", "Complex"],
            index=["Simple", "Medium", "Complex"].index(
                st.session_state.difficulty.capitalize()
            ),
            label_visibility="collapsed"
        )
        st.session_state.difficulty = difficulty.lower()
        
        diff_descriptions = {
            "Simple": "🌱 Basic recall & recognition questions",
            "Medium": "📚 Comprehension & application questions",
            "Complex": "🧠 Analysis & evaluation questions"
        }
        st.caption(diff_descriptions[difficulty])
    
    st.divider()
    
    if st.button("🎯 Generate Quiz", use_container_width=True, type="primary"):
        st.session_state.quiz_state = 'generating'
        st.rerun()


# ===== GENERATING SECTION =====
def render_generating_section():
    st.markdown("### 🤖 Generating Your Quiz")
    
    progress_bar = st.progress(0, text="Extracting text from slides...")
    time.sleep(0.5)
    
    progress_bar.progress(30, text="Analyzing content with AI...")
    time.sleep(0.5)
    
    progress_bar.progress(60, text="Generating questions & explanations...")
    
    try:
        slide_text = get_combined_text(st.session_state.file_id)
        
        questions = generate_with_fallback(
            slide_text,
            st.session_state.num_questions,
            st.session_state.difficulty
        )
        
        st.session_state.questions = questions
        st.session_state.current_q = 0
        st.session_state.answers = {}
        st.session_state.show_explanation = False
        st.session_state.quiz_submitted = False
        st.session_state.score = 0
        
        progress_bar.progress(100, text="Quiz generated successfully!")
        time.sleep(0.3)
        
        st.session_state.quiz_state = 'quiz'
        st.rerun()
        
    except Exception as e:
        st.markdown(f"""
        <div class="custom-error">❌ Failed to generate quiz: {str(e)}</div>
        """, unsafe_allow_html=True)
        
        if st.button("⬅️ Back to Configuration"):
            st.session_state.quiz_state = 'config'
            st.rerun()


# ===== QUIZ SECTION =====
def render_quiz_section():
    questions = st.session_state.questions
    total = len(questions)
    current = st.session_state.current_q
    q = questions[current]
    
    # Progress
    progress_val = (current + 1) / total
    st.progress(progress_val, text=f"Question {current + 1} of {total}")
    
    # Question card
    diff_class = f"badge-{st.session_state.difficulty}"
    
    st.markdown(f"""
    <div class="question-card">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <span class="question-number">Question {current + 1}</span>
            <span class="difficulty-badge {diff_class}">{st.session_state.difficulty.upper()}</span>
        </div>
        <p class="question-text">{q['question']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    options = q['options']
    option_labels = [opt['label'] for opt in options]
    option_texts = [f"{opt['label']}. {opt['text']}" for opt in options]
    
    # Check if already answered
    answered = str(current) in st.session_state.answers
    
    if answered and st.session_state.show_explanation:
        # Show options with correct/wrong highlights
        selected = st.session_state.answers[str(current)]
        for i, opt in enumerate(options):
            is_selected = opt['label'] == selected
            is_correct = opt['is_correct']
            
            if is_correct:
                st.success(f"**{opt['label']}.** {opt['text']} ✓")
            elif is_selected and not is_correct:
                st.error(f"**{opt['label']}.** {opt['text']} ✗")
            else:
                st.info(f"**{opt['label']}.** {opt['text']}")
        
        # Explanation
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">💡 Explanation</div>
            <div class="explanation-text">{}</div>
        </div>
        """.format(q['explanation']), unsafe_allow_html=True)
        
        # Distractor explanations
        if q.get('distractor_explanations'):
            st.markdown("##### Why the other options are wrong:")
            for label, explanation in q['distractor_explanations'].items():
                st.markdown(f"""
                <div style="padding:6px 0; border-bottom:1px solid #e2e8f0; font-size:0.85rem; color:#334155;">
                    <span style="font-weight:700; color:#dc2626;">{label}:</span> {explanation}
                </div>
                """, unsafe_allow_html=True)
        
        # Next/Submit button
        if current < total - 1:
            if st.button("Next Question →", use_container_width=True, type="primary"):
                st.session_state.current_q += 1
                st.session_state.show_explanation = False
                st.rerun()
        else:
            if st.button("📊 View Results", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = True
                st.session_state.quiz_state = 'results'
                st.rerun()
    
    else:
        # Show selectable options
        selected_option = st.radio(
            "Select your answer:",
            options=option_texts,
            key=f"q_{current}",
            label_visibility="collapsed",
            index=None
        )
        
        if selected_option:
            selected_label = selected_option.split(".")[0].strip()
            st.session_state.answers[str(current)] = selected_label
            
            # Check if correct
            correct_label = q['correct_answer']
            is_correct = selected_label == correct_label
            if is_correct:
                st.session_state.score += 1
            
            st.session_state.show_explanation = True
            st.rerun()


# ===== RESULTS SECTION =====
def render_results_section():
    questions = st.session_state.questions
    total = len(questions)
    score = st.session_state.score
    percentage = (score / total) * 100 if total > 0 else 0
    
    # Score circle
    st.markdown(f"""
    <div class="score-circle-container">
        <div class="score-circle">
            <svg class="score-circle-bg" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="65" fill="none" stroke="#e2e8f0" stroke-width="10"/>
                <circle cx="70" cy="70" r="65" fill="none" stroke="#10b981" stroke-width="10" 
                    stroke-dasharray="{2 * 3.14159 * 65}" stroke-dashoffset="{2 * 3.14159 * 65 * (1 - percentage/100)}" 
                    transform="rotate(-90, 70, 70)" stroke-linecap="round"/>
            </svg>
            <div class="score-circle-inner">
                <span class="score-big">{score}</span>
                <span class="score-total">/{total}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Results message
    if percentage >= 80:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h2 style="color:#0f172a; font-weight:700;">🎉 Excellent Work!</h2>
            <p style="color:#334155;">You have a strong understanding of the material!</p>
        </div>
        """, unsafe_allow_html=True)
    elif percentage >= 60:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h2 style="color:#0f172a; font-weight:700;">💪 Good Effort!</h2>
            <p style="color:#334155;">You're on the right track. Review the explanations below.</p>
        </div>
        """, unsafe_allow_html=True)
    elif percentage >= 40:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h2 style="color:#0f172a; font-weight:700;">📚 Keep Studying</h2>
            <p style="color:#334155;">Review the material and try again to improve your score.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h2 style="color:#0f172a; font-weight:700;">🔄 Let's Try Again</h2>
            <p style="color:#334155;">Review the material carefully and retake the quiz.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Score details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct", score)
    with col2:
        st.metric("Incorrect", total - score)
    with col3:
        st.metric("Accuracy", f"{percentage:.0f}%")
    
    st.divider()
    
    # Detailed breakdown
    st.markdown("### 📋 Detailed Review")
    
    for i, q in enumerate(questions):
        user_answer = st.session_state.answers.get(str(i), "Not answered")
        correct_answer = q['correct_answer']
        is_correct = user_answer == correct_answer
        
        result_class = "result-item-correct" if is_correct else "result-item-wrong"
        status_text = "✅ Correct" if is_correct else "❌ Incorrect"
        status_class = "correct" if is_correct else "wrong"
        
        # Get option texts
        correct_text = ""
        user_text = user_answer
        for opt in q['options']:
            if opt['label'] == correct_answer:
                correct_text = opt['text']
            if opt['label'] == user_answer:
                user_text = f"{opt['label']}. {opt['text']}"
        
        correct_display = f"{correct_answer}. {correct_text}" if correct_text else correct_answer
        
        st.markdown(f"""
        <div class="result-item {result_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.8rem; font-weight:600; color:#475569; text-transform:uppercase;">Question {i+1}</span>
                <span style="font-size:0.8rem; font-weight:600; padding:2px 10px; border-radius:12px; background:{'#d1fae5' if is_correct else '#fee2e2'}; color:{'#065f46' if is_correct else '#991b1b'}">{status_text}</span>
            </div>
            <p style="font-weight:600; color:#0f172a; margin-bottom:8px;">{q['question']}</p>
            <p style="font-size:0.85rem; color:#334155;"><strong>Your answer:</strong> {user_answer if user_answer != 'Not answered' else 'Not answered'}</p>
            <p style="font-size:0.85rem; color:#059669;"><strong>Correct answer:</strong> {correct_display}</p>
            <p style="font-size:0.85rem; color:#1e293b; margin-top:8px; padding-top:8px; border-top:1px solid #e2e8f0;">{q['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.current_q = 0
            st.session_state.answers = {}
            st.session_state.show_explanation = False
            st.session_state.quiz_submitted = False
            st.session_state.score = 0
            st.session_state.quiz_state = 'quiz'
            st.rerun()
    
    with col2:
        if st.button("📁 New Upload", use_container_width=True):
            # Clean up temp file
            if st.session_state.file_id and os.path.exists(st.session_state.file_id):
                os.unlink(st.session_state.file_id)
            
            # Reset all state
            for key in ['quiz_state', 'file_id', 'file_name', 'slide_count', 
                       'slides_preview', 'questions', 'current_q', 'answers',
                       'show_explanation', 'quiz_submitted', 'score']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ===== ROUTER =====
if st.session_state.quiz_state == 'upload':
    render_upload_section()
elif st.session_state.quiz_state == 'config':
    render_config_section()
elif st.session_state.quiz_state == 'generating':
    render_generating_section()
elif st.session_state.quiz_state == 'quiz':
    render_quiz_section()
elif st.session_state.quiz_state == 'results':
    render_results_section()

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid #e2e8f0; margin-top:2rem;">
    <p style="color:#475569; font-size:0.8rem;">Powered by OpenAI · Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)