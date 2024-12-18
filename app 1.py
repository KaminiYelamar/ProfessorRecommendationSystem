import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html file in your templates folder

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_excel('info_data (1).xlsx')
df = df.dropna(subset=['course'])

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Preprocess necessary columns
df['publications_processed'] = df['pub'].apply(preprocess_text)
df['course_name_processed'] = df['course name'].apply(preprocess_text)
df['course_processed'] = df['course'].apply(preprocess_text)
df['Keywords_processed'] = df['Keywords'].apply(preprocess_text)
df['no of responses'] = df['no of responses'].replace(0, 1)
df['adjusted_rating'] = df['Overall Summative Rating'] * (df['no of students'] / df['no of students'].max())
df['SFI'] = (df['adjusted_rating'] * df['Challenge and Engagement Index']) * (df['no of responses'] / df['no of students'])
df['PEI'] = (df['adjusted_rating'] + df['Challenge and Engagement Index']) * (df['no of responses'] / df['no of students'])
df['combined_features'] = df['course_processed'] + " " + df['course_name_processed'] + " " + df['publications_processed'] + " " + df['Keywords_processed']

# Recommendation function
def recommend_professors(text, course_code):
    filtered_professors = df[df['course'].str.contains(course_code, case=False)]
    filtered_professors['combined_features'].fillna('', inplace=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_professors['combined_features'])
    user_interests = " ".join(filter(lambda x: not x.startswith(course_code), text.split()))
    user_vector = tfidf_vectorizer.transform([user_interests])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    filtered_professors['Cosine Similarity'] = similarity_scores.flatten()

    recommended_professors = filtered_professors.sort_values(by=['Cosine Similarity'], ascending=[False])
    top_professors = []
    seen_professors = set()

    for _, row in recommended_professors.iterrows():
        if row['name'] not in seen_professors:
            seen_professors.add(row['name'])
            top_professors.append(row)
        if len(top_professors) > 4:
            break

    return pd.DataFrame(top_professors)[['name', 'course', 'Cosine Similarity', 'SFI', 'PEI']].sort_values(by=['SFI'], ascending=[False]).to_dict(orient="records")

# Flask route to handle recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    description = data.get('description', '')
    subject_code = data.get('subjectCode', '')
    try:
        recommendations = recommend_professors(description, subject_code)
        return jsonify({'recommendations': recommendations})
    except ValueError as e:
        if "empty vocabulary" in str(e):
            return jsonify({"error": "Invalid course ID or insufficient information. Please try a different input."}), 400
        else:
            return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
