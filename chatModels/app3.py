import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional
import json
import time

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="CineExtract AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    temperature=0
)

# --------------------------------------------------
# PYDANTIC MODEL
# --------------------------------------------------

class Movie(BaseModel):
    title: str
    release_date: Optional[int] = None
    genere: List[str]
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(
    pydantic_object=Movie
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the provided content.

        Return the information exactly according to the provided
        structured format.

        IMPORTANT:
        - Return only the structured JSON.
        - No explanations.
        - No markdown.
        - No additional text.

        {formatted_instructions}
        """
    ),
    (
        "human",
        "{movie_content}"
    )
])

chain = prompt | llm | parser


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(120, 70, 255, 0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(255, 50, 120, 0.10), transparent 30%),
        #08090d;
    color: #ffffff;
}

/* Hide default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hero */
.hero {
    text-align: center;
    padding: 40px 20px 30px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(140, 90, 255, 0.15);
    border: 1px solid rgba(160, 120, 255, 0.3);
    color: #bda7ff;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -2px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #bca5ff,
        #ff8ab8
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9a9ca8;
    font-size: 17px;
    margin-top: 14px;
}

/* Cards */
.card {
    background: rgba(20, 21, 29, 0.75);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Movie title */
.movie-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.movie-meta {
    color: #999daa;
    font-size: 14px;
}

/* Rating */
.rating {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 12px;
    background: rgba(255, 193, 7, 0.12);
    border: 1px solid rgba(255, 193, 7, 0.2);
    color: #ffd76a;
    font-weight: 700;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 6px 11px;
    margin: 4px 4px 4px 0;
    border-radius: 20px;
    background: rgba(130, 90, 255, 0.13);
    border: 1px solid rgba(140, 100, 255, 0.2);
    color: #c6b5ff;
    font-size: 12px;
}

/* Stats */
.stat {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
}

.stat-label {
    font-size: 12px;
    color: #858894;
    margin-top: 4px;
}

/* Text area */
textarea {
    background: #111219 !important;
    color: #eeeeee !important;
    border: 1px solid #292b36 !important;
    border-radius: 15px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #7957ff,
        #a855f7,
        #ec4899
    );
    color: white;
    font-weight: 700;
    font-size: 15px;
    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(150,80,255,0.35);
}

/* Divider */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 30px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #555864;
    font-size: 12px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown("""
<div class="hero">

<div class="hero-badge">
✦ AI POWERED MOVIE INTELLIGENCE
</div>

<h1>CineExtract AI</h1>

<p>
Turn raw movie information into structured, database-ready data.
</p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown("""
<div class="card">

<div class="section-title">
🎞️ Movie Content
</div>

</div>
""", unsafe_allow_html=True)

movie_text = st.text_area(
    "Movie content",
    height=230,
    label_visibility="collapsed",
    placeholder="""Paste movie information here...

Example:
3 Idiots is a 2009 Indian coming-of-age comedy-drama
directed by Rajkumar Hirani..."""
)

st.write("")

extract = st.button(
    "✨ Extract Movie Information"
)


# --------------------------------------------------
# EXTRACTION
# --------------------------------------------------

if extract:

    if not movie_text.strip():

        st.warning("Please enter some movie content first.")

    else:

        with st.spinner("Analyzing movie content..."):

            start = time.time()

            try:

                movie = chain.invoke({
                    "movie_content": movie_text,
                    "formatted_instructions":
                        parser.get_format_instructions()
                })

                elapsed = time.time() - start

                st.success(
                    f"Extraction completed in {elapsed:.2f} seconds"
                )

                st.markdown("<div class='divider'></div>",
                            unsafe_allow_html=True)

                # --------------------------------------------------
                # MOVIE HEADER
                # --------------------------------------------------

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(
                        f"""
                        <div class="movie-title">
                        {movie.title}
                        </div>

                        <div class="movie-meta">
                        🎬 {movie.director or "Director unavailable"}
                        &nbsp;&nbsp;•&nbsp;&nbsp;
                        📅 {movie.release_date or "Year unavailable"}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    if movie.rating:

                        st.markdown(
                            f"""
                            <div style="text-align:right">
                            <span class="rating">
                            ⭐ {movie.rating}
                            </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.write("")

                # --------------------------------------------------
                # STATS
                # --------------------------------------------------

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown(
                        f"""
                        <div class="stat">
                        <div class="stat-number">
                        {len(movie.cast or [])}
                        </div>
                        <div class="stat-label">
                        CAST MEMBERS
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="stat">
                        <div class="stat-number">
                        {len(movie.genere)}
                        </div>
                        <div class="stat-label">
                        GENRES
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown(
                        f"""
                        <div class="stat">
                        <div class="stat-number">
                        {movie.release_date or "—"}
                        </div>
                        <div class="stat-label">
                        RELEASE YEAR
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c4:
                    st.markdown(
                        f"""
                        <div class="stat">
                        <div class="stat-number">
                        {movie.rating or "—"}
                        </div>
                        <div class="stat-label">
                        RATING
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")
                st.write("")

                # --------------------------------------------------
                # GENRE
                # --------------------------------------------------

                st.markdown(
                    "<div class='section-title'>🎭 Genres</div>",
                    unsafe_allow_html=True
                )

                tags = ""

                for genre in movie.genere:
                    tags += f'<span class="tag">{genre}</span>'

                st.markdown(tags, unsafe_allow_html=True)

                st.write("")

                # --------------------------------------------------
                # MAIN CONTENT
                # --------------------------------------------------

                left, right = st.columns([1, 1])

                with left:

                    st.markdown(
                        """
                        <div class="card">
                        <div class="section-title">
                        👥 Cast
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if movie.cast:

                        for actor in movie.cast:
                            st.markdown(
                                f"• {actor}"
                            )

                    else:
                        st.write("Not available")

                    st.markdown("</div>",
                                unsafe_allow_html=True)

                with right:

                    st.markdown(
                        """
                        <div class="card">
                        <div class="section-title">
                        📝 Summary
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(movie.summary)

                    st.markdown("</div>",
                                unsafe_allow_html=True)

                # --------------------------------------------------
                # RAW JSON
                # --------------------------------------------------

                st.write("")
                st.write("")

                st.markdown(
                    "<div class='section-title'>"
                    "⚡ Database-Ready JSON"
                    "</div>",
                    unsafe_allow_html=True
                )

                movie_json = movie.model_dump()

                st.json(movie_json)

                st.download_button(
                    label="⬇️ Download JSON",
                    data=json.dumps(
                        movie_json,
                        indent=4
                    ),
                    file_name="movie.json",
                    mime="application/json"
                )

            except Exception as e:

                st.error(
                    "Extraction failed. Please check the model response."
                )

                with st.expander("Technical details"):
                    st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
CineExtract AI &nbsp;•&nbsp; Built with LangChain + NVIDIA + Pydantic
</div>
""", unsafe_allow_html=True)