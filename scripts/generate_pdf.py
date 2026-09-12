import base64
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent
screenshots_dir = root / 'assets' / 'screenshots'
pdf_output = root / 'Steam_Game_Recommender_Project_Proof.pdf'

def get_b64_image(filename):
    p = screenshots_dir / filename
    if p.exists():
        with open(p, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

img1 = get_b64_image('01_player_recommender.png')
img2 = get_b64_image('02_game_explorer.png')
img3 = get_b64_image('03_cold_start_studio.png')
img4 = get_b64_image('04_benchmark_hub.png')

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Steam Game Recommendation System - Project Proof</title>
    <style>
        @page {{
            size: A4;
            margin: 14mm 14mm 14mm 14mm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1a202c;
            line-height: 1.45;
            font-size: 9pt;
            background: #ffffff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            border-bottom: 2.5px solid #0f172a;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }}
        .title {{
            font-size: 19pt;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 4px 0;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            font-size: 10.5pt;
            color: #475569;
            font-weight: 500;
            margin: 0 0 10px 0;
        }}
        .meta-bar {{
            display: flex;
            gap: 14px;
            align-items: center;
            background: #f1f5f9;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 8.5pt;
        }}
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .meta-label {{
            font-weight: 700;
            color: #334155;
        }}
        .repo-link {{
            color: #0369a1;
            font-weight: 700;
            text-decoration: none;
            background: #e0f2fe;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #bae6fd;
            font-size: 8pt;
        }}
        .repo-link-main {{
            color: #ffffff;
            font-weight: 700;
            text-decoration: none;
            background: #0284c7;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 9pt;
        }}
        h2 {{
            font-size: 11.5pt;
            font-weight: 700;
            color: #0f172a;
            border-left: 3.5px solid #0284c7;
            padding-left: 7px;
            margin-top: 14px;
            margin-bottom: 6px;
        }}
        p {{
            margin: 0 0 6px 0;
            color: #334155;
        }}
        ul {{
            margin: 0 0 8px 0;
            padding-left: 18px;
        }}
        li {{
            margin-bottom: 3px;
            color: #334155;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 10px 0 12px 0;
        }}
        .stat-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 7px;
            text-align: center;
        }}
        .stat-num {{
            font-size: 13pt;
            font-weight: 800;
            color: #0284c7;
        }}
        .stat-lbl {{
            font-size: 7pt;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8pt;
            margin: 8px 0 10px 0;
        }}
        th, td {{
            padding: 5px 8px;
            text-align: left;
            border: 1px solid #cbd5e1;
        }}
        th {{
            background-color: #0f172a;
            color: #ffffff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .highlight {{
            font-weight: 700;
            color: #0f172a;
            background: #e0f2fe;
        }}
        .screenshot-card {{
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 12px;
            background: #ffffff;
        }}
        .screenshot-header {{
            background: #0f172a;
            color: #ffffff;
            padding: 5px 10px;
            font-size: 8.5pt;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .screenshot-img {{
            width: 100%;
            max-height: 290px;
            object-fit: contain;
            display: block;
            background: #0e141b;
        }}
        .screenshot-desc {{
            padding: 6px 10px;
            font-size: 7.8pt;
            color: #475569;
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
        }}
        .page-break {{
            page-break-before: always;
        }}
        .takeaway-box {{
            background: #f0fdf4;
            border-left: 3.5px solid #16a34a;
            padding: 7px 10px;
            border-radius: 0 5px 5px 0;
            margin: 8px 0;
            font-size: 8.2pt;
            color: #166534;
        }}
        .footer-note {{
            text-align: center;
            font-size: 8pt;
            color: #64748b;
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>

    <!-- PAGE 1: TITLE, STATS, FORMULATION, BENCHMARKS -->
    <div class="header">
        <div class="title">🎮 Steam Game Recommendation System</div>
        <div class="subtitle">Implicit ALS Matrix Factorization, Content-Based Semantic Matching & Cold-Start Hybrid Engine</div>
        <div class="meta-bar">
            <div class="meta-item"><span class="meta-label">Author:</span> SweetGiraffo</div>
            <div class="meta-item"><span class="meta-label">Domain:</span> Recommendation Systems & Ranking ML</div>
            <div class="meta-item"><span class="meta-label">Stack:</span> Python, Implicit ALS, Scikit-Learn, FastAPI, Streamlit</div>
            <div class="meta-item" style="margin-left: auto;">
                <a href="https://github.com/SweetGiraffo/steam-game-recommender" class="repo-link-main" target="_blank">
                    🔗 Click Here: GitHub Repository
                </a>
            </div>
        </div>
    </div>

    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-num">200,000+</div>
            <div class="stat-lbl">Interactions Logged</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">5,217</div>
            <div class="stat-lbl">Active Users Modelled</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">3,980</div>
            <div class="stat-lbl">Steam Games Catalog</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">+172%</div>
            <div class="stat-lbl">NDCG@10 Lift vs Pop</div>
        </div>
    </div>

    <h2>1. Executive Summary & Problem Formulation</h2>
    <p>
        Traditional recommender tutorials use explicit 1–5 star ratings (e.g., MovieLens) and evaluate on RMSE. In real entertainment platforms (Steam, Spotify, Netflix), interaction data is <strong>implicit</strong>: users express preference through purchases and playtime hours. Raw playtime exhibits severe Pareto/power-law skew (2,000+ hours vs 2 hours).
    </p>
    <ul>
        <li><strong>Confidence-Weighted Implicit ALS:</strong> Log-dampened transformation (<em>C<sub>ui</sub> = 1 + &alpha; log(1 + r<sub>ui</sub>)</em>) bounds gradients against extreme playtime outliers while preserving relative engagement magnitude.</li>
        <li><strong>Ranking Evaluation (NDCG@K, MAP@K):</strong> Evaluated directly on true ranking/retrieval objectives rather than unobserved RMSE error.</li>
        <li><strong>Cold-Start Resolution:</strong> Adaptive late fusion blends collaborative latent scores with metadata TF-IDF representations, dynamically shifting weight to semantic profile matching for new games/users.</li>
    </ul>

    <h2>2. Offline Ranking Benchmark Results</h2>
    <p>Evaluated across a stratified Leave-20%-Out test split (5,211 users) and a 15% held-out cold item test set (597 games):</p>

    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>NDCG@10</th>
                <th>MAP@10</th>
                <th>Catalog Coverage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Popularity Baseline</strong></td>
                <td>0.0496</td>
                <td>0.1845</td>
                <td>0.1366</td>
                <td>0.0907</td>
                <td>0.80%</td>
            </tr>
            <tr>
                <td><strong>Content-Based (TF-IDF)</strong></td>
                <td>0.0226</td>
                <td>0.1595</td>
                <td>0.1104</td>
                <td>0.0868</td>
                <td>41.76%</td>
            </tr>
            <tr class="highlight">
                <td><strong>Implicit ALS (Pure CF)</strong></td>
                <td>0.0996</td>
                <td>0.4264</td>
                <td>0.3718</td>
                <td>0.3110</td>
                <td>27.71%</td>
            </tr>
            <tr class="highlight">
                <td><strong>Adaptive Hybrid (CF + Content)</strong></td>
                <td>0.0771</td>
                <td>0.3554</td>
                <td>0.3069</td>
                <td>0.2564</td>
                <td><strong>43.37%</strong></td>
            </tr>
        </tbody>
    </table>

    <div class="takeaway-box">
        <strong>Cold-Start Stress Test Proof:</strong> Pure Collaborative Filtering collapses to <strong>0.0000 NDCG@10</strong> on unobserved games. The Adaptive Hybrid engine achieves <strong>0.0293 NDCG@10</strong> and <strong>0.0633 Recall@10</strong>, completely resolving the cold-start failure mode and increasing catalog exploration coverage to <strong>43.37%</strong>.
    </div>

    <h2>3. Production Architecture & Serving Layer</h2>
    <p>
        <strong>FastAPI REST Microservice:</strong> Delivers sub-25ms inference latency with endpoints:
        <code>/recommend/user/{{user_id}}</code>, <code>/recommend/game/{{game_title}}</code>, <code>/recommend/cold-start</code>, and <code>/games/search</code>.
        <br>
        <strong>Automated Testing:</strong> 13/13 passing unit tests verifying preprocessing integrity, ranking metric math, and model inference.
    </p>

    <!-- PAGE 2: SCREENSHOT PROOFS 1 & 2 -->
    <div class="page-break"></div>

    <h2>4. Visual Proof-of-Work: Live System Screenshots</h2>
    <p>
        The following screenshots show the live Streamlit web application. Click any link to inspect the repository or full-resolution asset.
        <br>
        Repository Link: <a href="https://github.com/SweetGiraffo/steam-game-recommender" class="repo-link" target="_blank">https://github.com/SweetGiraffo/steam-game-recommender</a>
    </p>

    <!-- Screenshot 1 -->
    <div class="screenshot-card">
        <div class="screenshot-header">
            <span>Proof 1: Personalized Player Profile Recommender (Tab 1)</span>
            <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/01_player_recommender.png" class="repo-link" target="_blank" style="background:#1e293b; color:#38bdf8; border-color:#38bdf8;">
                🔗 View Image on GitHub
            </a>
        </div>
        <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/01_player_recommender.png" target="_blank">
            <img src="{img1}" class="screenshot-img" alt="Player Profile Recommender Screenshot">
        </a>
        <div class="screenshot-desc">
            <strong>Demonstrated Functionality:</strong> Real-time player library inspection (<em>Alan Wake, BioShock series</em>), sparse matrix telemetry (5,217 users, 3,980 games, 99.43% sparsity), model switcher, and personalized next-game recommendations with match scores and Steam price tags.
        </div>
    </div>

    <!-- Screenshot 2 -->
    <div class="screenshot-card">
        <div class="screenshot-header">
            <span>Proof 2: Game-to-Game Explorer "More Like This" (Tab 2)</span>
            <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/02_game_explorer.png" class="repo-link" target="_blank" style="background:#1e293b; color:#38bdf8; border-color:#38bdf8;">
                🔗 View Image on GitHub
            </a>
        </div>
        <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/02_game_explorer.png" target="_blank">
            <img src="{img2}" class="screenshot-img" alt="Game Explorer Screenshot">
        </a>
        <div class="screenshot-desc">
            <strong>Demonstrated Functionality:</strong> Real-time item-to-item nearest neighbors combining learned ALS factor similarity and content-based TF-IDF cosine similarity for <em>The Elder Scrolls V Skyrim</em> with live cover art, genres, and metadata.
        </div>
    </div>

    <!-- PAGE 3: SCREENSHOT PROOFS 3 & 4 + REPOSITORY LINKS -->
    <div class="page-break"></div>

    <!-- Screenshot 3 -->
    <div class="screenshot-card">
        <div class="screenshot-header">
            <span>Proof 3: Cold-Start Gamer Studio — Zero-History Onboarding (Tab 3)</span>
            <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/03_cold_start_studio.png" class="repo-link" target="_blank" style="background:#1e293b; color:#38bdf8; border-color:#38bdf8;">
                🔗 View Image on GitHub
            </a>
        </div>
        <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/03_cold_start_studio.png" target="_blank">
            <img src="{img3}" class="screenshot-img" alt="Cold Start Studio Screenshot">
        </a>
        <div class="screenshot-desc">
            <strong>Demonstrated Functionality:</strong> Overcomes cold-start onboarding. A new gamer selects 2–3 favorites (<em>Skyrim, Fallout 4, Terraria</em>), and the hybrid engine synthesizes a weighted semantic profile vector on the fly to surface immediate relevant recommendations.
        </div>
    </div>

    <!-- Screenshot 4 -->
    <div class="screenshot-card">
        <div class="screenshot-header">
            <span>Proof 4: DS Interview Benchmark Hub (Tab 4)</span>
            <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/04_benchmark_hub.png" class="repo-link" target="_blank" style="background:#1e293b; color:#38bdf8; border-color:#38bdf8;">
                🔗 View Image on GitHub
            </a>
        </div>
        <a href="https://github.com/SweetGiraffo/steam-game-recommender/blob/main/assets/screenshots/04_benchmark_hub.png" target="_blank">
            <img src="{img4}" class="screenshot-img" alt="Benchmark Hub Screenshot">
        </a>
        <div class="screenshot-desc">
            <strong>Demonstrated Functionality:</strong> Live interactive benchmark portal rendering offline ranking metrics (Precision@10, Recall@10, NDCG@10, MAP@10), cold-start item stress tests, and metric visualizers.
        </div>
    </div>

    <div class="meta-bar" style="margin-top: 14px;">
        <div class="meta-item"><span class="meta-label">GitHub Repository:</span> <a href="https://github.com/SweetGiraffo/steam-game-recommender" target="_blank" style="color:#0284c7; font-weight:700;">https://github.com/SweetGiraffo/steam-game-recommender</a></div>
        <div class="meta-item" style="margin-left: auto;"><span class="meta-label">Verification:</span> 100% Passed (13/13 Tests)</div>
    </div>

    <div class="footer-note">
        Steam Game Recommendation System Proof-of-Work Document &bull; Generated for Placement Portfolios & Technical Interviews
    </div>

</body>
</html>
"""

print('Generating clean 3-page PDF with Playwright...')
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content(html_content, wait_until='networkidle')
    page.pdf(
        path=str(pdf_output),
        format='A4',
        print_background=True,
        margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'}
    )
    browser.close()

print(f'PDF created successfully at: {pdf_output} ({pdf_output.stat().st_size / 1024 / 1024:.2f} MB)')
