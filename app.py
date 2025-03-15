from flask import Flask, render_template, request
import pickle
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the saved LDA model and dictionary
lda_model = LdaModel.load("lda_model.pkl")
dictionary = Dictionary.load("dictionary.pkl")

# Preprocess the text input
def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Get LDA topics and their word counts
def get_lda_topics(model, corpus, num_topics=5):
    topics = model.get_document_topics(corpus)
    topic_words = []
    topic_word_count = []
    
    # For each topic in the document
    for topic_id, topic_prob in topics:
        words = model.show_topic(topic_id, topn=50)  # Get the top words for each topic
        topic_words.append([word for word, _ in words])
        
        # Calculate word count based on BoW
        word_count = sum([count for word_id, count in corpus if dictionary[word_id] in dict(words)])
        topic_word_count.append(word_count)
    
    return topic_words, topic_word_count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input text from the form
        text = request.form['text_input']
        
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        
        # Convert the cleaned text into a corpus format for LDA
        bow_vector = dictionary.doc2bow(cleaned_text.split())
        
        # Get the topics and their word counts for the input text
        topics, topic_word_count = get_lda_topics(lda_model, bow_vector)
        
        return render_template('index.html', topics=topics, topic_word_count=topic_word_count)
    
    return render_template('index.html', topics=[], topic_word_count=[])

if __name__ == "__main__":
    app.run(debug=True)
