from flask import Flask, render_template, request
import pickle
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import re
from nltk.corpus import stopwords
import nltk
import traceback  # For detailed error logging

nltk.download('stopwords')  # Download stopwords if not already downloaded

app = Flask(__name__)

# Load the saved LDA model and dictionary
lda_model = LdaModel.load("lda_model.pkl")  # Load the LDA model from a pickle file
dictionary = Dictionary.load("dictionary.pkl")  # Load the dictionary

# Preprocess the text input (cleaning, removing stopwords, etc.)
def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))  # Define English stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to get LDA topics and word counts
def get_lda_topics(model, corpus, num_topics=5):
    try:
        topics = model.get_document_topics(corpus)  # Get the topic distribution for the document
        if not topics:
            raise ValueError("No topics were generated from the input text.")

        topic_words = []
        topic_word_counts = []
        
        for topic_id, _ in topics:
            words = model.show_topic(topic_id, topn=10)  # Get the top 10 words for each topic
            if not words:
                raise ValueError(f"No words found for topic {topic_id}.")
            topic_words.append([word for word, _ in words])  # Append words for each topic
            topic_word_counts.append(len(words))  # Count of words per topic

        return topic_words, topic_word_counts

    except Exception as e:
        # Log the detailed traceback for debugging
        print(f"Error in get_lda_topics: {e}")
        traceback.print_exc()  # Prints the detailed traceback in the server logs
        return [], []  # Return empty lists if an error occurs

# Route for the home page
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

            # Convert the cleaned text into a corpus format (BoW) for LDA
            bow_vector = dictionary.doc2bow(cleaned_text.split())

            # Get the topics and word counts for the input text
            topics, topic_word_counts = get_lda_topics(lda_model, bow_vector)

            # Check if topics and word counts were returned properly
            if not topics or not topic_word_counts:
                raise ValueError("Failed to generate topics or word counts.")

            # Combine topics and word counts using zip()
            topics_with_counts = list(zip(topics, topic_word_counts))

            return render_template('index.html', topics=topics_with_counts, error=None)

        except Exception as e:
            # If an error occurs, log it and show an error message
            print(f"Error processing text: {e}")
            traceback.print_exc()  # Prints the detailed traceback in the server logs
            return render_template('index.html', topics=[], error="An error occurred while processing the text. Please check the input or server logs for details.")
    
    # For a GET request, render the form with no topics
    return render_template('index.html', topics=[])

# Main entry point for the app
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for easier debugging during development
