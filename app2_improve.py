# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import requests
import streamlit as st

# proyerktin csv ve pkl fayllarinin oldugu qovluq
PATH = "/Users/dinarabbasli/Desktop/project movie rec"
TMDB_API_KEY = "270f5a1752652e85edc33b1e5bc813f6"  # boş saxlanan ppster yeirne placholder gosterilir
PLACEHOLDER_POSTER = "https://placehold.co/300x445/1c1c24/e5e5e5?text=No+Poster"

# #data ve modelleri cache-leyirik ki her kliklede yeniden oxunmasin
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path + "/" + "movie_data_for_app.csv")
    dataframe = pd.read_csv(file_path + "/" + "movie_dataframe_for_app.csv")
    return data, dataframe

## cashe dta ve resource ist etmisem ki user her klikleynde deyisende nesese agir fayl ve metrisler 
#yendien yaddasa yuklenmesin
@st.cache_resource
def load_models(file_path):
    sig = joblib.load(file_path + "/" + "sigmoid_kernel.pkl")
    tfv = joblib.load(file_path + "/" + "tfidf_vectorizer.pkl")
    return tfv, sig


data, dataframe = load_data(PATH)
tfv, sig = load_models(PATH)


# esas tovsiye funksiyasi - jupyterde yazdigim menteqin eynisi, sadece Streamlit-e uygun return edir
def give_recommendations(movie_title, model, data, dataframe, top_n=10):
    indices = pd.Series(data=data.index, index=data["original_title"])
    idx = indices[movie_title]

    model_scores = list(enumerate(model[idx]))
    model_scores_sorted = sorted(model_scores, key=lambda x: x[1], reverse=True)
    model_scores_10 = model_scores_sorted[1: top_n + 1] 
    ##ilk yeri skip edirem cunku film oz-ozune 100% oxsayir(oci indexs filmin ozudur)

    movie_indices = [i[0] for i in model_scores_10]
    scores = [i[1] for i in model_scores_10]

    result = dataframe.iloc[movie_indices].copy()
    result["match_score"] = scores
    return result.reset_index(drop=True)

##TMDB- saytindan poster axtarir, api key yoxdursa ve ya tapilmasa None qaytarir
@st.cache_data(show_spinner=False) ##yene chaching eger user 2 defe eyni film secse kod tmdb saytina
##sorgu gonderib vaxt itirmesin
def fetch_poster(title, api_key):

    if not api_key: ##api key var ay yox
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": title},
            timeout=4,
        )
        results = r.json().get("results")
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w342" + results[0]["poster_path"]
    except Exception:
        pass
    return None


##sehife ayarlari-baslq hissesi-barda
st.set_page_config(
    page_title="Simple Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

##burda css yazmisam ki app kinoteatr kimi tund fonda gorunsun, defolt streamlit deyil

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #1b1b24 0%, #0f0f14 60%);
            color: #eaeaf0;
        }
        h1, h2, h3, .stMarkdown p { color: #eaeaf0 !important; }

        .hero {
            padding: 1.2rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #e5303040, #7a1fa240);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: 0.35rem 0 0 0; opacity: 0.85; }

        div[data-testid="stSelectbox"] label { font-weight: 600; opacity: 0.9; }

        .stButton > button {
            background: linear-gradient(135deg, #e53030, #b3164f);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(229, 48, 48, 0.35);
        }

        .movie-card {
            background: #1a1a22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            overflow: hidden;
            transition: transform 0.15s ease, border-color 0.15s ease;
            height: 100%;
        }
        .movie-card:hover {
            transform: translateY(-4px);
            border-color: rgba(229, 48, 48, 0.55);
        }
        .movie-card img { width: 100%; display: block; }
        .movie-card-body { padding: 0.7rem 0.8rem 0.9rem 0.8rem; }
        .movie-rank {
            display: inline-block;
            background: #e53030;
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            margin-bottom: 0.4rem;
        }
        .movie-title {
            font-weight: 700;
            font-size: 0.95rem;
            line-height: 1.25rem;
            margin: 0 0 0.25rem 0;
            min-height: 2.5rem;
        }
        .movie-meta {
            font-size: 0.78rem;
            opacity: 0.7;
            margin: 0;
        }
        footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
##basliq
st.markdown(
    """
    <div class="hero">
        <h1>🎬 Simple Movie Recommender</h1>
        <p>Sevdiyin filmə bənzər filmləri kəşf et — Content-based filtering ilə</p>
    </div>
    """,
    unsafe_allow_html=True,
)
###secim ve duyme
col_select, col_button = st.columns([4, 1]) ## ekrani 4:1 nisbetde bol 2 yere

movie_list = data["original_title"].sort_values().tolist()
with col_select:
    selected_movie = st.selectbox("Bir film seç:", movie_list)

with col_button:
    st.write("")  # duymeni selectbox ilə üfüqi düzlendirmək ucun boşluq
    st.write("")
    go = st.button("Tövsiyələri göstər", use_container_width=True)

##duyme basilanda tovsiyeleri gosteririk
if go:
    if not selected_movie:
        st.warning("Zəhmət olmasa əvvəlcə bir film seçin.")
    else:
        with st.spinner("Bənzər filmlər axtarılır..."):#model filmeri hesablayan ekranda cixan
            recommendations = give_recommendations(selected_movie, sig, data, dataframe)
           #sig matrixde filmler oxsairgi salanilirdir(model),movie title-selected movie selectboxdaki
        st.subheader(f"'{selected_movie}' filminə bənzər filmlər")

        extra_cols = [
            c for c in ["genres", "release_date", "vote_average"]
            if c in recommendations.columns
        ]

        cols_per_row = 5 ##her sirada 5 fiilm olsun
        rows = [
            recommendations.iloc[i: i + cols_per_row]
            for i in range(0, len(recommendations), cols_per_row)
        ]

        for row_df in rows: ##filmi yerine oturmtaq ucun
            cols = st.columns(cols_per_row)
            for col, (_, movie_row) in zip(cols, row_df.iterrows()):
                title = movie_row["original_title"]
                rank = movie_row.name + 1

                poster_url = fetch_poster(title, TMDB_API_KEY) or PLACEHOLDER_POSTER

                meta_bits = []
                if "release_date" in extra_cols and pd.notna(movie_row.get("release_date")):
                    meta_bits.append(str(movie_row["release_date"])[:4])
                    ##ILDEN 4 REQEMI GOTUTURUK FILMIN CIXMA TARIXINDEN- mes 2010
                if "weighted_avg" in extra_cols and pd.notna(movie_row.get("weighted_avg")):
                    #meta_bits.append(f"⭐ {movie_row['weighted_avg']}")
                    meta_bits.append(f"⭐ {round(movie_row['weighted_avg'], 1)}")
                    ## bu da imdbsidir filmelrin biz hesbaladigimiz
                meta_line = " · ".join(meta_bits)

                with col:
                    st.markdown( #visual kart kimi edirik
                        f"""
                        <div class="movie-card">
                            <img src="{poster_url}" />
                            <div class="movie-card-body">
                                <span class="movie-rank">#{rank}</span>
                                <p class="movie-title">{title}</p>
                                <p class="movie-meta">{meta_line}</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
