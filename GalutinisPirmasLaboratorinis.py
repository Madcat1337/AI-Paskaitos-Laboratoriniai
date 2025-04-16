import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import os
from flask import Flask, request, render_template_string

# --------------------- CLEAN TEXT ---------------------
def clean_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[âÂ€Ã©œ™¦;]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

# --------------------- TOKENIZE & GET TOP WORDS ---------------------
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    words = text.lower().split()
    words = [word for word in words if len(word) > 2 and not word.isdigit()]
    return words

def get_top_words(text, top_n=5):
    tokens = simple_tokenize(text)
    stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'was', 'for',
                  'on', 'are', 'as', 'with', 'they', 'be', 'at', 'this', 'have', 'from'}
    tokens = [word for word in tokens if word not in stop_words]
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(top_n)
    return ', '.join([word for word, _ in most_common])

# --------------------- SMART CSV READER ---------------------
def read_and_prepare_csv(csv_path):
    print(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, engine='python', encoding='utf-8', on_bad_lines='skip')
        if df.shape[1] == 1:
            first_row = df.iloc[0, 0]
            if ';' in first_row:
                df = pd.read_csv(csv_path, sep=';', engine='python', encoding='utf-8', on_bad_lines='skip')
            elif ',' in first_row:
                df = pd.read_csv(csv_path, sep=',', engine='python', encoding='utf-8', on_bad_lines='skip')
            elif '\t' in first_row:
                df = pd.read_csv(csv_path, sep='\t', engine='python', encoding='utf-8', on_bad_lines='skip')
            else:
                raise ValueError("Unknown separator.")
        if df.shape[1] < 4:
            raise ValueError(f"CSV format error: Expected at least 4 columns, got {df.shape[1]}.")
        df.columns = ['count', 'headline', 'info', 'is_fake']
        return df
    except Exception as e:
        print("Failed to read CSV:", e)
        raise

# --------------------- MAIN ANALYSIS ---------------------
def analyze_articles(csv_path, export_path=None):
    if export_path is None:
        export_path = os.path.splitext(csv_path)[0] + "_analyzed.csv"

    df = read_and_prepare_csv(csv_path)
    df['headline_clean'] = df['headline'].apply(clean_text)
    df['info_clean'] = df['info'].apply(clean_text)
    df['combined_text'] = df['headline_clean'] + ' ' + df['info_clean']

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df['combined_text'])
    feature_names = vectorizer.get_feature_names_out()

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=30,
        max_iter=600,
        init='k-means++'
    )
    df['cluster'] = kmeans.fit_predict(X)

    cluster_keywords = {}
    cluster_labels = {}
    for cluster in range(5):
        cluster_docs = df[df['cluster'] == cluster]
        if len(cluster_docs) > 0:
            cluster_X = X[cluster_docs.index]
            cluster_word_scores = cluster_X.mean(axis=0).A1
            top_words = sorted(
                zip(feature_names, cluster_word_scores),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            bigrams = [w for w, s in top_words if ' ' in w][:1]
            cluster_labels[cluster] = bigrams[0] if bigrams else top_words[0][0]
            cluster_keywords[cluster] = [(word, round(score * 100, 2)) for word, score in top_words if score > 0]

    df['top_keywords'] = df['combined_text'].apply(lambda x: get_top_words(x))
    export_columns = ['count', 'headline_clean', 'info_clean', 'is_fake', 'cluster', 'top_keywords']
    df[export_columns].to_csv(export_path, index=False)

    return df, cluster_keywords, cluster_labels, vectorizer, X

# --------------------- FLASK APP ---------------------
app = Flask(__name__)
analysis_data = {
    'df': None,
    'cluster_keywords': None,
    'cluster_labels': None,
    'vectorizer': None,
    'X': None
}

@app.route('/', methods=['GET', 'POST'])
def index():
    global analysis_data
    search_results = []
    search_query = request.form.get('search', '').lower() if request.method == 'POST' else ''

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        df, cluster_keywords, cluster_labels, vectorizer, X = analyze_articles(file_path)
        analysis_data.update({
            'df': df,
            'cluster_keywords': cluster_keywords,
            'cluster_labels': cluster_labels,
            'vectorizer': vectorizer,
            'X': X
        })

    if search_query and analysis_data['df'] is not None:
        df = analysis_data['df']
        mask = (df['headline_clean'].str.lower().str.contains(search_query, na=False) |
                df['info_clean'].str.lower().str.contains(search_query, na=False))
        # Get full text and highlight search phrase
        search_results = [
            {
                'text': re.sub(
                    f'({re.escape(search_query)})',
                    r'<span class="highlight">\1</span>',
                    row['combined_text'],
                    flags=re.IGNORECASE
                ),
                'keywords': row['top_keywords']
            }
            for _, row in df[mask].iterrows()
        ][:5]  # Limit to 5 results

    return render_template_string(
        HTML_TEMPLATE,
        cluster_keywords=analysis_data['cluster_keywords'],
        cluster_labels=analysis_data['cluster_labels'],
        search_results=search_results,
        has_data=analysis_data['df'] is not None,
        search_query=search_query
    )

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>News Analyzer Pro</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            text-align: center;
        }
        .upload-form, .search-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        input[type="file"], input[type="text"] {
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
            width: 70%;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background: #00d4ff;
            border: none;
            border-radius: 5px;
            color: #1a1a2e;
            cursor: pointer;
            transition: all 0.3s;
        }
        input[type="submit"]:hover {
            background: #00b7d4;
            transform: translateY(-2px);
        }
        .cluster {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #00d4ff;
        }
        .cluster h3 {
            color: #00d4ff;
            margin-top: 0;
        }
        .cluster ul {
            list-style: none;
            padding: 0;
        }
        .cluster li {
            padding: 5px 0;
            color: #e0e0e0;
        }
        .cluster li:before {
            content: "• ";
            color: #00d4ff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        th {
            background: rgba(0, 212, 255, 0.2);
            color: #fff;
        }
        .no-results {
            text-align: center;
            padding: 20px;
            color: #ff6b6b;
        }
        .highlight {
            background-color: #00d4ff;
            color: #1a1a2e;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .article-text {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>News Analyzer Pro</h1>
        <form class="upload-form" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv">
            <input type="submit" value="Analyze Articles">
        </form>
        
        {% if has_data %}
            <div class="search-box">
                <form method="post">
                    <input type="text" name="search" placeholder="Search article content..." value="{{ search_query }}">
                    <input type="submit" value="Search">
                </form>
            </div>
        {% endif %}

        {% if cluster_keywords and cluster_labels %}
            <h2>Article Themes</h2>
            {% for cluster, keywords in cluster_keywords.items() %}
                <div class="cluster">
                    <h3>{{ cluster_labels[cluster] }} (Theme {{ cluster + 1 }})</h3>
                    <ul>
                        {% for word, score in keywords %}
                            <li>{{ word }} (Score: {{ score }})</li>
                        {% endfor %}
                    </ul>
                </div>
            {% endfor %}
        {% endif %}

        {% if search_results %}
            <h2>Search Results (Max 5)</h2>
            <table>
                <tr><th>Full Article</th><th>Key Phrases</th></tr>
                {% for result in search_results %}
                    <tr>
                        <td class="article-text">{{ result.text | safe }}</td>
                        <td>{{ result.keywords }}</td>
                    </tr>
                {% endfor %}
            </table>
        {% else %}
            {% if has_data and search_results is not none %}
                <p class="no-results">No articles found matching your search.</p>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

# --------------------- RUN ---------------------
if __name__ == "__main__":
    app.run(debug=True)