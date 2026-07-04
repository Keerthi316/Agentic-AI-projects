"""AI Content Engine — Streamlit application entry point."""

import streamlit as st

import config
from image_gen import generate_image
from text_gen import (
    generate_blog_intro,
    generate_image_prompt,
    generate_social_posts,
    generate_tagline,
)
from utils import save_campaign, timestamped_filename, word_count
from video_gen import build_video_prompt, generate_video

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Engine",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 AI Content Engine")
st.caption("Generate a full marketing campaign from a single product brief.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Campaign Brief")

    product = st.text_input("Product Name", placeholder="e.g. AquaFlow Smart Bottle")
    audience = st.text_input("Target Audience", placeholder="e.g. Health-conscious millennials")
    tone = st.selectbox("Brand Tone", config.BRAND_TONES)

    st.divider()
    generate_btn = st.button("✨ Generate Campaign", type="primary", use_container_width=True)



def _validate() -> bool:
    if not product.strip():
        st.sidebar.error("Please enter a product name.")
        return False
    if not audience.strip():
        st.sidebar.error("Please enter a target audience.")
        return False
    missing = config.validate_keys()
    if missing:
        st.sidebar.error(f"Missing API keys: {', '.join(missing)}")
        return False
    return True


# ── Main columns ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

if generate_btn:
    if not _validate():
        st.stop()

    tone_style = config.TONE_STYLE_MAP[tone]

    # ── STEP 1: Tagline ───────────────────────────────────────────────────────
    with left:
        with st.container(border=True):
            st.subheader("🏷️ Campaign Tagline")
            with st.spinner("Crafting tagline…"):
                try:
                    tagline = generate_tagline(product, audience, tone)
                except Exception as exc:
                    st.error(f"Tagline generation failed: {exc}")
                    st.stop()
            st.markdown(f"## *{tagline}*")

        # ── STEP 2: Blog Intro ────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("📝 Blog Introduction")
            with st.spinner("Writing blog introduction…"):
                try:
                    blog = generate_blog_intro(product, audience, tone, tagline)
                except Exception as exc:
                    st.error(f"Blog generation failed: {exc}")
                    st.stop()
            st.write(blog)
            st.caption(f"Word count: {word_count(blog)}")

        # ── STEP 3: Social Posts ──────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("📱 Social Media Posts")
            with st.spinner("Writing social posts…"):
                try:
                    posts = generate_social_posts(product, audience, tone, tagline)
                except Exception as exc:
                    st.error(f"Social post generation failed: {exc}")
                    st.stop()

            st.markdown("**𝕏 Twitter**")
            st.info(posts.get("twitter", ""))
            st.markdown("**📸 Instagram**")
            st.info(posts.get("instagram", ""))
            st.markdown("**💼 LinkedIn**")
            st.info(posts.get("linkedin", ""))

    # ── STEP 4: Hero Image ────────────────────────────────────────────────────
    with right:
        with st.container(border=True):
            st.subheader("🖼️ Hero Image")
            with st.spinner("Generating hero image (this may take ~30s)…"):
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
            with st.spinner("Generating video (this may take 1–3 min)…"):
                try:
                    video_prompt = build_video_prompt(product, tone)
                    vid_filename = timestamped_filename("promo", "mp4")
                    vid_path = generate_video(img_path, video_prompt, filename=vid_filename)
                except Exception as exc:
                    st.error(f"Video generation failed: {exc}")
                    st.info(
                        "💡 Sora video generation requires approved API access. "
                        "If you don't have access yet, the campaign content above is still complete."
                    )
                    st.stop()
            st.video(vid_path)

    # ── Persist campaign ──────────────────────────────────────────────────────
    try:
        json_path = save_campaign(
            product, tagline, blog, posts,
            img_path, vid_path if "vid_path" in dir() else "",
        )
        st.success(f"✅ Campaign saved → `{json_path}`")
    except Exception:
        pass

else:
    with left:
        st.info("Fill in the brief and click **✨ Generate Campaign** to get started.")
    with right:
        st.info("Your hero image and promo video will appear here.")
