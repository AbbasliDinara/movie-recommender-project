# 🎬 Movie Recommender System / Film Tövsiyə Sistemi

## 🇬🇧 English

A movie recommendation engine built using Content-based Filtering to discover movies similar to any selected title.

### ⚙️ How It Works
Movie plot summaries (*overviews*) are vectorized using **TF-IDF**, and pairwise similarity scores are calculated via **Sigmoid Kernel**. Once a movie is selected, the system retrieves and displays the top 10 most similar titles.

**Key Features:**
- Implements **IMDb's Weighted Average** formula to score and rank movies fairly based on ratings and vote counts.
- Includes an experimental **SVM (Support Vector Machine)** model trained to predict a movie's genre directly from its overview text.

### 📁 File Structure
- `notebook.ipynb` — Data preprocessing, TF-IDF vectorization, Sigmoid Kernel implementation, Weighted Average scoring, and SVM model training.
- `app.py` — Web application built with Streamlit for selecting movies and viewing real-time recommendations.

### 🛠️ Tech Stack
Python, pandas, scikit-learn, Streamlit

<img width="2828" height="1542" alt="image" src="https://github.com/user-attachments/assets/5bfea796-2e03-417a-a7d2-ba87f957e799" />
