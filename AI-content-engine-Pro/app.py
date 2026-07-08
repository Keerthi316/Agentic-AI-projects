"""AI Content Engine Pro — Streamlit application entry point.

New in Pro:
  • AI Self-Critique Loop  (critic.py)   — auto-evaluates and regenerates failing assets
  • Voiceover Generation   (voice_gen.py) — blog → adapted script → MP3 audio player
  • Multi-Channel Adaptation (adaptation.py) — rewrite text for LinkedIn / TikTok / Facebook
  • Better UX: input validation, granular progress bar, graceful API error handling
"""

from __future__ import annotations

import streamlit as st

import config
from adaptation import CHANNEL_OPTIONS, adapt_for_channel
from critic import CriticResult, critique_blog, critique_social_posts, critique_tagline
from image_gen import generate_image
from text_gen import (
    generate_blog_intro,
    generate_image_prompt,
    generate_social_posts,
    generate_tagline,
)
from utils import save_campaign, timestamped_filename, word_count
from video_gen import build_video_prompt, generate_video
from voice_gen import generate_voiceover

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Engine Pro",
    page_icon="🚀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .verdict-pass  { background:#d4edda; color:#155724; border-radius:6px; padding:6px 10px; font-size:0.85rem; }
    .verdict-warn  { background:#fff3cd; color:#856404; border-radius:6px; padding:6px 10px; font-size:0.85rem; }
    .verdict-fail  { background:#f8d7da; color:#721c24; border-radius:6px; padding:6px 10px; font-size:0.85rem; }
    .channel-badge { background:#e8f4fd; color:#0c5460; border-radius:6px; padding:4px 8px;
                     font-size:0.8rem; font-weight:600; display:inline-block; margin-bottom:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 AI Content Engine Pro")
st.caption("Generate a full marketing campaign — with AI critique, voiceover, and multi-channel adaptation.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎯 Campaign Brief")

    product = st.text_input(
        "Product Name",
        placeholder="e.g. AquaFlow Smart Bottle",
        help="The product you want to promote.",
    )
    audience = st.text_input(
        "Target Audience",
        placeholder="e.g. Health-conscious millennials",
        help="Who are you trying to reach?",
    )
    tone = st.selectbox(
        "Brand Tone",
        config.BRAND_TONES,
        help="Sets the visual and copy style for the entire campaign.",
    )

    st.divider()

    st.subheader("⚙️ Pro Options")
    run_critic = st.toggle(
        "AI Self-Critique Loop",
        value=True,
        help="Evaluate each text asset and regenerate if it fails quality checks (adds ~15 s).",
    )
    run_voiceover = st.toggle(
        "Generate Voiceover",
        value=True,
        help="Synthesise an MP3 voiceover from the final approved blog intro.",
    )

    st.divider()
    generate_btn = st.button("✨ Generate Campaign", type="primary", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate() -> bool:
    """Validate sidebar inputs and API keys. Writes errors to sidebar. Returns True if OK."""
    ok = True
    if not product.strip():
        st.sidebar.error("⚠️ Please enter a product name.")
        ok = False
    if not audience.strip():
        st.sidebar.error("⚠️ Please enter a target audience.")
        ok = False
    missing = config.validate_keys()
    if missing:
        st.sidebar.error(f"⚠️ Missing API keys:\n" + "\n".join(f"• {m}" for m in missing))
        ok = False
    return ok


def _verdict_badge(result: CriticResult) -> str:
    """Return an HTML badge for the critic verdict."""
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    icon = icons.get(result.verdict, "ℹ️")
    css_class = f"verdict-{result.verdict}"
    label = result.verdict.upper()
    notes = result.notes[:100] + "…" if len(result.notes) > 100 else result.notes
    return f'<span class="{css_class}">{icon} Critic: {label} — {notes}</span>'


def _show_critic_result(result: CriticResult) -> None:
    """Render critic badge + optional warnings in the current Streamlit context."""
    st.markdown(_verdict_badge(result), unsafe_allow_html=True)
    if result.attempts > 1:
        st.caption(f"↺ Regenerated {result.attempts - 1}× based on critic feedback.")
    if result.warnings:
        with st.expander("⚠️ Critic warnings", expanded=False):
            for w in result.warnings:
                st.warning(w)


def _show_social_posts(tw: str, ig: str, li: str) -> None:
    """Render the three social media posts with platform labels."""
    st.markdown("**𝕏 Twitter / X**")
    st.info(tw)
    st.markdown("**📸 Instagram**")
    st.info(ig)
    st.markdown("**💼 LinkedIn**")
    st.info(li)


# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

if generate_btn:
    if not _validate():
        st.stop()

    tone_style = config.TONE_STYLE_MAP[tone]

    # Total progress steps:
    #  1=tagline, 2=tagline-critic, 3=blog, 4=blog-critic,
    #  5=social, 6=social-critic, 7=image, 8=video, 9=voiceover
    total_steps = 9 if run_voiceover else 8
    if not run_critic:
        total_steps -= 3
    progress = st.progress(0, text="Starting campaign generation…")
    _step = [0]  # mutable container to allow mutation from nested function

    def _advance(label: str) -> None:
        _step[0] += 1
        progress.progress(min(_step[0] / total_steps, 1.0), text=f"⏳ {label}")

    # ── STEP 1: Tagline ───────────────────────────────────────────────────────
    with left:
        with st.container(border=True):
            st.subheader("🏷️ Campaign Tagline")
            _advance("Crafting tagline…")
            with st.spinner("Crafting tagline…"):
                try:
                    tagline = generate_tagline(product, audience, tone)
                except Exception as exc:
                    st.error(f"Tagline generation failed: {exc}")
                    st.stop()

            if run_critic:
                _advance("Critic evaluating tagline…")
                with st.spinner("Critic evaluating tagline…"):
                    try:
                        tagline_result = critique_tagline(tagline, product, audience, tone)
                        tagline = tagline_result.final_text
                    except Exception as exc:
                        st.warning(f"Critic unavailable for tagline: {exc}")
                        tagline_result = None
            else:
                tagline_result = None

            st.markdown(f"## *{tagline}*")
            if tagline_result:
                _show_critic_result(tagline_result)

        # ── STEP 2: Blog Intro ────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("📝 Blog Introduction")
            _advance("Writing blog introduction…")
            with st.spinner("Writing blog introduction…"):
                try:
                    blog = generate_blog_intro(product, audience, tone, tagline)
                except Exception as exc:
                    st.error(f"Blog generation failed: {exc}")
                    st.stop()

            if run_critic:
                _advance("Critic evaluating blog…")
                with st.spinner("Critic evaluating blog…"):
                    try:
                        blog_result = critique_blog(blog, product, audience, tone, tagline)
                        blog = blog_result.final_text
                    except Exception as exc:
                        st.warning(f"Critic unavailable for blog: {exc}")
                        blog_result = None
            else:
                blog_result = None

            st.write(blog)
            st.caption(f"Word count: {word_count(blog)}")
            if blog_result:
                _show_critic_result(blog_result)

        # ── STEP 3: Social Posts ──────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("📱 Social Media Posts")
            _advance("Writing social posts…")
            with st.spinner("Writing social posts…"):
                try:
                    posts = generate_social_posts(product, audience, tone, tagline)
                except Exception as exc:
                    st.error(f"Social post generation failed: {exc}")
                    st.stop()

            if run_critic:
                _advance("Critic evaluating social posts…")
                with st.spinner("Critic evaluating social posts…"):
                    try:
                        posts_result = critique_social_posts(posts, product, audience, tone, tagline)
                        # Retrieve the dict that the critic wrapper stashed on the result
                        posts = getattr(posts_result, "final_posts_dict", posts)
                    except Exception as exc:
                        st.warning(f"Critic unavailable for social posts: {exc}")
                        posts_result = None
            else:
                posts_result = None

            _show_social_posts(
                posts.get("twitter", ""),
                posts.get("instagram", ""),
                posts.get("linkedin", ""),
            )
            if posts_result:
                _show_critic_result(posts_result)

    # ── STEP 4: Hero Image ────────────────────────────────────────────────────
    with right:
        with st.container(border=True):
            st.subheader("🖼️ Hero Image")
            _advance("Generating hero image (this may take ~30 s)…")
            with st.spinner("Generating hero image (this may take ~30 s)…"):
                try:
                    img_prompt = generate_image_prompt(product, audience, tone, tone_style)
                    img_filename = timestamped_filename("hero", "png")
                    img_path = generate_image(img_prompt, filename=img_filename)
                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")
                    st.stop()
            st.image(img_path, use_column_width=True)
            with st.expander("Image prompt used"):
                st.caption(img_prompt)

        # ── STEP 5: Promo Video ───────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("🎬 Promotional Video")
            _advance("Generating video (this may take 1–3 min)…")
            vid_path = ""
            with st.spinner("Generating video (this may take 1–3 min)…"):
                try:
                    video_prompt = build_video_prompt(product, tone)
                    vid_filename = timestamped_filename("promo", "mp4")
                    vid_path = generate_video(img_path, video_prompt, filename=vid_filename)
                    st.video(vid_path)
                except Exception as exc:
                    st.error(f"Video generation failed: {exc}")
                    st.info(
                        "💡 Sora/Wan video generation requires approved API access. "
                        "The campaign content above is still complete."
                    )

    # ── STEP 6: Voiceover ─────────────────────────────────────────────────────
    if run_voiceover:
        _advance("Generating voiceover…")
        st.divider()
        st.subheader("🎙️ Blog Voiceover")
        with st.spinner("Adapting blog to voice script and synthesising audio…"):
            try:
                voice_script, audio_path = generate_voiceover(blog, product, tone)
                vo_col1, vo_col2 = st.columns([2, 1])
                with vo_col1:
                    with st.expander("📄 Adapted Voice Script", expanded=True):
                        st.write(voice_script)
                with vo_col2:
                    st.markdown("**▶️ Audio Playback**")
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download MP3",
                            data=f,
                            file_name="voiceover.mp3",
                            mime="audio/mp3",
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(f"Voiceover generation failed: {exc}")

    # ── STEP 7: Multi-Channel Adaptation ─────────────────────────────────────
    st.divider()
    st.subheader("📡 Multi-Channel Adaptation")
    st.caption(
        "Rewrite text assets (tagline, blog, social posts) for a specific distribution channel. "
        "Hero image and video remain unchanged."
    )

    channel = st.selectbox(
        "Select Distribution Channel",
        CHANNEL_OPTIONS,
        index=0,
        help="Choose a channel to rewrite the text assets for.",
        key="channel_selector",
    )

    adapt_btn = st.button(
        f"🔄 Adapt for {channel}",
        type="secondary",
        disabled=(channel == "Original"),
        key="adapt_btn",
    )

    if channel == "Original":
        st.info("Select a channel other than 'Original' to rewrite the assets.")

    if adapt_btn and channel != "Original":
        with st.spinner(f"Rewriting assets for {channel}…"):
            try:
                adapted = adapt_for_channel(
                    channel=channel,
                    product=product,
                    audience=audience,
                    tone=tone,
                    tagline=tagline,
                    blog=blog,
                    posts=posts,
                )

                st.markdown(
                    f'<span class="channel-badge">📡 {channel}</span>',
                    unsafe_allow_html=True,
                )

                adapt_left, adapt_right = st.columns(2, gap="large")
                with adapt_left:
                    with st.container(border=True):
                        st.subheader("🏷️ Adapted Tagline")
                        st.markdown(f"## *{adapted['tagline']}*")
                    with st.container(border=True):
                        st.subheader("📝 Adapted Blog Introduction")
                        st.write(adapted["blog"])
                        st.caption(f"Word count: {word_count(adapted['blog'])}")
                with adapt_right:
                    with st.container(border=True):
                        st.subheader("📱 Adapted Social Posts")
                        _show_social_posts(
                            adapted["twitter"],
                            adapted["instagram"],
                            adapted["linkedin"],
                        )
            except Exception as exc:
                st.error(f"Channel adaptation failed: {exc}")

    # ── Persist campaign ──────────────────────────────────────────────────────
    progress.progress(1.0, text="✅ Campaign complete!")
    try:
        json_path = save_campaign(
            product, tagline, blog, posts, img_path, vid_path
        )
        st.success(f"✅ Campaign saved → `{json_path}`")
    except Exception:
        pass  # Non-fatal — don't interrupt the user experience

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    with left:
        st.info(
            "👈 Fill in the Campaign Brief in the sidebar and click "
            "**✨ Generate Campaign** to get started."
        )
        st.markdown("**What you'll get:**")
        st.markdown(
            "- 🏷️ Campaign Tagline\n"
            "- 📝 Blog Introduction (~200 words)\n"
            "- 📱 Social posts for Twitter, Instagram & LinkedIn\n"
            "- 🤖 AI self-critique & auto-refinement *(Pro)*\n"
            "- 🎙️ Voiceover MP3 from blog *(Pro)*\n"
            "- 📡 Multi-channel adaptation *(Pro)*"
        )
    with right:
        st.info("Your hero image and promo video will appear here.")
        st.markdown("**Supported channels:**")
        st.markdown(
            "- B2B LinkedIn — professional, ROI-focused\n"
            "- Gen-Z TikTok — casual, emoji-heavy, viral\n"
            "- Parents Facebook — warm, family-centric"
        )
