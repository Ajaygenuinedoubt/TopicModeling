from flask import Flask, render_template, request
import pickle
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download stopwords if not already present
nltk.download('stopwords')

# Load the saved LDA model and dictionary
try:
    lda_model = LdaModel.load("lda_model.pkl")
    dictionary = Dictionary.load("dictionary.pkl")
except Exception as e:
    print(f"Error loading model or dictionary: {e}")

# Preprocess the text input
def preprocess_text(text):
    try:
        text = re.sub(r'\n', ' ', text)  # Remove newlines
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        raise e  # Raise the error to the main try-except block

# Get LDA topics and word count per topic
def get_lda_topics(model, corpus, num_topics=5):
    try:
        topics = model.get_document_topics(corpus)
        topic_words = []
        topic_word_counts = []
        for topic_id, _ in topics:
            words = model.show_topic(topic_id, topn=50)  # Get the top words for each topic
            word_list = [word for word, _ in words]
            topic_words.append(word_list)
            # Count word occurrences for each topic
            word_count = sum([corpus.count(dictionary.token2id[word]) for word in word_list if word in dictionary.token2id])
            topic_word_counts.append(word_count)
        return topic_words, topic_word_counts
    except Exception as e:
        print(f"Error in generating topics: {e}")
        raise e

# Index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the input text from the form
            text = request.form['text_input']

            # Check if the text is empty
            if not text.strip():
                return render_template('index.html', topics=[], error="Text input cannot be empty.")
            
            # Preprocess the text
            cleaned_text = preprocess_text(text)

            # Convert the cleaned text into a corpus format for LDA
            bow_vector = dictionary.doc2bow(cleaned_text.split())

            # Get the topics and word counts for the input text
            topics, topic_word_counts = get_lda_topics(lda_model, bow_vector)

            # Render the template with topics and word counts
            return render_template('index.html', topics=topics, word_counts=topic_word_counts)

        except Exception as e:
            # Log the exception and return a user-friendly error message
            print(f"Error processing text: {e}")
            return render_template('index.html', topics=[], error="An error occurred while processing the text. Please check the input or server logs for details.")
    
    # For GET request, simply render the form
    return render_template('index.html', topics=[])

# Error handling for 500 internal server errors
@app.errorhandler(500)
def internal_error(error):
    return "500 error: An internal error occurred. Please try again.", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
