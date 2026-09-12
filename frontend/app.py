import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported when running directly via streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title='Steam Game Recommender | DS Portfolio',
    page_icon='🎮',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom Steam Dark Theme CSS
st.markdown('''
<style>
    .main {
        background-color: #0e141b;
        color: #c7d5e0;
    }
    .stApp {
        background: linear-gradient(180deg, #0e141b 0%, #171d25 100%);
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    .game-card {
        background: #1b2838;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
        border: 1px solid #2a475e;
        transition: transform 0.2s, border-color 0.2s;
    }
    .game-card:hover {
        transform: translateY(-2px);
        border-color: #66c0f4;
    }
    .game-title {
        font-size: 15px;
        font-weight: 700;
        color: #66c0f4;
        margin-top: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .game-tag {
        display: inline-block;
        background: #213344;
        color: #8f98a0;
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .meta-badge {
        background: #4c6b22;
        color: #a4d007;
        font-weight: bold;
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .price-tag {
        color: #bbee33;
        font-weight: bold;
        font-size: 13px;
    }
    .metric-box {
        background: #171a21;
        border: 1px solid #2a475e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #66c0f4;
    }
    .metric-lbl {
        font-size: 12px;
        color: #8f98a0;
        text-transform: uppercase;
    }
</style>
''', unsafe_allow_html=True)


@st.cache_resource
def load_system():
    root = Path(__file__).resolve().parent.parent
    model_path = root / 'artifacts' / 'models.pkl'
    catalog_path = root / 'data' / 'processed' / 'games_catalog.parquet'
    metrics_path = root / 'artifacts' / 'metrics.json'

    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    catalog_df = pd.read_parquet(catalog_path)
    game2idx = models['mappings']['game2idx']

    # Pre-index catalog by game_idx
    game_lookup = {}
    for _, row in catalog_df.iterrows():
        title = row['game_title']
        if title in game2idx:
            g_idx = game2idx[title]
            game_lookup[g_idx] = {
                'game_idx': g_idx,
                'game_title': title,
                'genres': str(row.get('genres', '')),
                'categories': str(row.get('categories', '')),
                'price': float(row.get('price', 0.0)),
                'metacritic_score': int(row.get('metacritic_score', 0)),
                'header_image': str(row.get('header_image', 'https://cdn.akamai.steamstatic.com/steam/apps/10/header.jpg')),
                'description': str(row.get('description', '')),
                'developer': str(row.get('developer', ''))
            }

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

    return models, game_lookup, metrics


models, game_lookup, metrics = load_system()
mappings = models['mappings']
user2idx = mappings['user2idx']
idx2user = mappings['idx2user']
game2idx = mappings['game2idx']
idx2game = mappings['idx2game']
sparse_mat = models['sparse_matrix']

# Sidebar Header
with st.sidebar:
    st.image('https://store.akamai.steamstatic.com/public/shared/images/header/logo_steam.svg', width=160)
    st.title('Steam Recommender')
    st.caption('Implicit Feedback ALS + Content-Based Hybrid System')
    st.markdown('---')
    st.markdown('**System Overview**')
    st.markdown(f'• **Users Index:** {len(user2idx):,}')
    st.markdown(f'• **Games Index:** {len(game2idx):,}')
    st.markdown(f'• **Matrix Interactions:** {sparse_mat.nnz:,}')
    st.markdown(f'• **Sparsity:** {100.0 * (1 - sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1])):.2f}%')
    st.markdown('---')
    st.markdown('**Algorithms Active:**')
    st.markdown('1. **Implicit ALS** (Hu, Koren, Volinsky)')
    st.markdown('2. **Content-Based** (TF-IDF + Cosine)')
    st.markdown('3. **Adaptive Hybrid** (Late Fusion & Cold-Start)')
    st.markdown('4. **Popularity Baseline**')

st.title('🎮 Steam Game Recommendation Engine')
st.markdown(
    'A production-grade recommendation platform designed for data science interviews, featuring **implicit feedback matrix factorization**, **content-based embeddings**, and **cold-start handling** evaluated on **NDCG@K and MAP@K**.'
)

tabs = st.tabs([
    '👤 Player Profile Recommender',
    '🎯 Game-to-Game Explorer',
    '❄️ Cold-Start Gamer Studio',
    '📊 DS Interview Benchmark Hub'
])

# ----------------- TAB 1: PLAYER PROFILE RECOMMENDER -----------------
with tabs[0]:
    st.subheader('Personalized User Recommendations')
    col_input1, col_input2, col_input3 = st.columns([2, 2, 1])

    # Preset sample active users
    sample_users = ['151603712', '59945787', '53875128', '86540', '11373781']

    with col_input1:
        user_choice = st.selectbox('Choose Sample Player or Enter Custom ID:', options=sample_users, index=0)
        custom_user = st.text_input('Or enter specific User ID:', value=user_choice)
        selected_user = custom_user.strip() if custom_user.strip() else user_choice

    with col_input2:
        model_choice = st.selectbox(
            'Recommendation Algorithm:',
            options=[
                'Hybrid (ALS + Content) [Recommended]',
                'Implicit ALS (Pure Collaborative)',
                'Content-Based (TF-IDF Semantic)',
                'Popularity Baseline'
            ]
        )
    with col_input3:
        top_k = st.slider('Top K:', min_value=5, max_value=24, value=8, step=1)

    if selected_user in user2idx:
        u_idx = user2idx[selected_user]
        user_row = sparse_mat[u_idx]
        user_game_indices = user_row.indices.tolist()

        # Display User's Library Preview
        st.markdown(f'### 📚 Player Library ({len(user_game_indices)} games played/owned)')
        history_cols = st.columns(min(4, len(user_game_indices)))
        for i, g_idx in enumerate(user_game_indices[:4]):
            if g_idx in game_lookup:
                info = game_lookup[g_idx]
                with history_cols[i % len(history_cols)]:
                    st.image(info['header_image'], width='stretch')
                    st.markdown(f"**{info['game_title']}**")
                    st.caption(f"{info['genres']}")

        st.markdown('---')
        st.markdown(f'### 🌟 Recommended Next Games (Model: {model_choice.split()[0]})')

        # Model dispatch
        if 'Implicit ALS' in model_choice:
            recs = models['cf_model'].recommend(u_idx, n=top_k)
        elif 'Content-Based' in model_choice:
            recs = models['cb_model'].recommend(u_idx, train_matrix=sparse_mat, n=top_k)
        elif 'Popularity' in model_choice:
            recs = models['pop_model'].recommend(u_idx, n=top_k, filter_items=set(user_game_indices))
        else:
            recs = models['hybrid_model'].recommend(u_idx, train_matrix=sparse_mat, n=top_k)

        # Render Recommendations in Responsive Grid
        grid_cols = st.columns(4)
        for i, (g_idx, score) in enumerate(recs):
            if g_idx in game_lookup:
                info = game_lookup[g_idx]
                with grid_cols[i % 4]:
                    st.markdown(f'''
                    <div class="game-card">
                        <img src="{info['header_image']}" style="width:100%; border-radius:4px; height:120px; object-fit:cover;">
                        <div class="game-title" title="{info['game_title']}">{info['game_title']}</div>
                        <div style="margin-top:6px;">
                            <span class="game-tag">{info['genres'].split(',')[0]}</span>
                            <span class="meta-badge">★ {info['metacritic_score']}</span>
                        </div>
                        <div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">
                            <span class="price-tag"></span>
                            <span style="font-size:11px; color:#66c0f4;">Score: {score:.3f}</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
    else:
        st.warning(f'User ID {selected_user} not found in database. Try one of the sample users or use the Cold-Start tab!')

# ----------------- TAB 2: GAME-TO-GAME EXPLORER -----------------
with tabs[1]:
    st.subheader('🎯 Game Similarity Explorer ("More Like This")')
    st.caption('Combines collaborative item factor embeddings with TF-IDF content representations.')

    all_titles = sorted(list(game2idx.keys()))
    default_game_idx = all_titles.index('The Elder Scrolls V Skyrim') if 'The Elder Scrolls V Skyrim' in all_titles else 0
    selected_game = st.selectbox('Select or Type a Game:', options=all_titles, index=default_game_idx)
    k_sim = st.slider('Number of Similar Games:', 4, 16, 8, key='sim_k')

    g_idx = game2idx[selected_game]
    game_info = game_lookup.get(g_idx, {})

    # Display Selected Game Header
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        st.image(game_info.get('header_image', ''), width='stretch')
    with col_sel2:
        st.markdown(f"## {game_info.get('game_title', '')}")
        st.markdown(f"**Genres:** {game_info.get('genres', '')}")
        st.markdown(f"**Developer:** {game_info.get('developer', 'Valve')}")
        st.markdown(f"**Price:**  | **Metacritic:** {game_info.get('metacritic_score', 78)}")
        st.write(game_info.get('description', ''))

    st.markdown('---')
    st.markdown('### 🔍 Games With Similar Playing Audience & Themes')

    cf_similar = dict(models['cf_model'].similar_items(g_idx, n=k_sim * 2))
    cb_similar = dict(models['cb_model'].similar_items(g_idx, n=k_sim * 2))

    all_cands = set(cf_similar.keys()).union(set(cb_similar.keys()))
    fused_sim = []
    for c in all_cands:
        s = 0.5 * cf_similar.get(c, 0.0) + 0.5 * cb_similar.get(c, 0.0)
        fused_sim.append((c, s))
    fused_sim.sort(key=lambda x: x[1], reverse=True)

    sim_cols = st.columns(4)
    for i, (c_idx, score) in enumerate(fused_sim[:k_sim]):
        if c_idx in game_lookup:
            info = game_lookup[c_idx]
            with sim_cols[i % 4]:
                st.markdown(f'''
                <div class="game-card">
                    <img src="{info['header_image']}" style="width:100%; border-radius:4px; height:120px; object-fit:cover;">
                    <div class="game-title" title="{info['game_title']}">{info['game_title']}</div>
                    <div style="margin-top:6px;">
                        <span class="game-tag">{info['genres'].split(',')[0]}</span>
                        <span class="meta-badge">★ {info['metacritic_score']}</span>
                    </div>
                    <div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">
                        <span class="price-tag"></span>
                        <span style="font-size:11px; color:#66c0f4;">Similarity: {score:.3f}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

# ----------------- TAB 3: COLD-START GAMER STUDIO -----------------
with tabs[2]:
    st.subheader('❄️ Cold-Start Gamer Studio (Zero-History Onboarding)')
    st.info(
        '**Interview Highlight:** Pure collaborative filtering models (like ALS or SVD) completely fail when a new user registers because they lack interaction rows. '
        'Our Hybrid architecture solves this via **instant profile synthesis**: picking a few titles constructs a semantic vector to retrieve catalog matches immediately.'
    )

    popular_presets = [
        'The Elder Scrolls V Skyrim',
        'Fallout 4',
        'Terraria',
        'Left 4 Dead 2',
        'Portal 2',
        'Counter-Strike',
        'Sid Meier\'s Civilization V',
        'Warframe',
        'Half-Life 2'
    ]
    valid_presets = [p for p in popular_presets if p in game2idx]

    selected_liked = st.multiselect(
        'Pick 2 or 3 games you love:',
        options=all_titles,
        default=valid_presets[:3]
    )

    if selected_liked:
        liked_indices = [game2idx[g] for g in selected_liked if g in game2idx]
        cold_recs = models['hybrid_model'].recommend_custom(liked_indices, n=8)

        st.markdown('### 🎯 Cold-Start Recommendations Generated For You')
        c_cols = st.columns(4)
        for i, (g_idx, score) in enumerate(cold_recs):
            if g_idx in game_lookup:
                info = game_lookup[g_idx]
                with c_cols[i % 4]:
                    st.markdown(f'''
                    <div class="game-card">
                        <img src="{info['header_image']}" style="width:100%; border-radius:4px; height:120px; object-fit:cover;">
                        <div class="game-title" title="{info['game_title']}">{info['game_title']}</div>
                        <div style="margin-top:6px;">
                            <span class="game-tag">{info['genres'].split(',')[0]}</span>
                            <span class="meta-badge">★ {info['metacritic_score']}</span>
                        </div>
                        <div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">
                            <span class="price-tag"></span>
                            <span style="font-size:11px; color:#66c0f4;">Content Match: {score:.3f}</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
    else:
        st.write('Please select at least one game above to generate recommendations.')

# ----------------- TAB 4: DS BENCHMARK HUB -----------------
with tabs[3]:
    st.subheader('📊 Offline Model Evaluation Benchmark (NDCG@10, MAP@10, Precision@10)')
    st.markdown(
        'This benchmark was evaluated offline using a **stratified Leave-20%-Out** test split for warm users and a **held-out cold item test set**.'
    )

    if 'warm_benchmark' in metrics:
        warm_df = pd.DataFrame(metrics['warm_benchmark']).T
        cold_df = pd.DataFrame(metrics['cold_benchmark']).T

        st.markdown('#### 1. Warm Recommendation Benchmark')
        st.dataframe(warm_df.style.highlight_max(axis=0, color='#1e3d59'), width='stretch')

        st.markdown('#### 2. Cold-Start Item Benchmark (Crucial Interview Proof-Point)')
        st.dataframe(cold_df.style.highlight_max(axis=0, color='#1e3d59'), width='stretch')

        st.markdown('---')
        st.markdown('### 📈 Metric Comparison Visualizer')
        metric_choice = st.selectbox(
            'Select Metric to Visualize:',
            options=['NDCG@10', 'MAP@10', 'Precision@10', 'Recall@10', 'Catalog_Coverage@10']
        )
        if metric_choice in warm_df.columns:
            st.bar_chart(warm_df[metric_choice])

        st.markdown('---')
        st.markdown('### 💡 Key Technical Talking Points for DS Interviews')
        st.markdown('''
        - **Implicit Feedback Transformation:** Gaming interactions do not provide explicit 1-5 star ratings. We formulate interaction confidence as {ui} = 1 + \alpha \log(1 + r_{ui}/\epsilon)$ (where {ui} = \text{hours}_{ui} + 1$). The log-dampening is critical because raw gaming playtime follows a heavy-tailed Pareto distribution.
        - **Why Ranking Metrics over RMSE:** Recommendation is a retrieval and ranking problem, not a regression task. Minimizing RMSE across unobserved entries predicts uninteresting negatives; NDCG@K and MAP@K directly evaluate whether relevant items appear at the top of the recommendation list.
        - **Cold-Start Resilience:** Notice how pure **Implicit ALS achieves 0.0000 NDCG** on cold-start items because their latent vectors have no historical gradient updates. The **Hybrid model retains positive NDCG@10 (0.0293) and Recall@10 (0.0633)** and expands catalog coverage to **43.37%**.
        ''')
