from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

app = Flask(__name__)

# Load the trained model
model = load_model('sentiment_model.h5')

# Load the CountVectorizer
with open('count_vectorizer.pickle', 'rb') as handle:
    count_vectorizer = pickle.load(handle)

# Load the LabelEncoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

translator = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()

def remove_emoji(text):
    # Regular expression to remove emojis
    return re.sub(r'[^\w\s,]', '', text)

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match
    
def preprocess_text(text):
    # Remove emojis
    text = remove_emoji(text)
    
    # Tokenize
    tokens = word_tokenize(text.lower())  # Lowercasing and tokenizing
    
    # Remove punctuation
    tokens = [word.translate(translator) for word in tokens]
    
    # PoS Tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Map PoS tags to WordNet tags
    pos_tags = [(word, pos_tagger(tag)) for word, tag in pos_tags]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos) for word, pos in pos_tags]
    
    return ' '.join(lemmatized_tokens)

# route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML form to take comments as input

# route to handle sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the comment from the form
        comment = request.form['comment']
        
        if not comment:
            return jsonify({'error': 'No comment provided'}), 400
        
        # Preprocess the comment
        tokens = preprocess_text(comment)

        vectorized_comment = count_vectorizer.transform([tokens])
    
        
        # Predict using the model
        prediction = model.predict(vectorized_comment.toarray())
        
        predicted_class_index = np.argmax(prediction, axis=1)[0]
    
        # mapping of class indices to sentiment labels
        class_labels = ['negative', 'neutral', 'positive']  
        sentiment = class_labels[predicted_class_index]
        
        # Return the result
        return render_template('result.html', comment=comment, sentiment=sentiment)
    
    except Exception as e:
        # Return an error message in case of error
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
