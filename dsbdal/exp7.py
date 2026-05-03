import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK data is available
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ==========================================
# FIX: Added 'averaged_perceptron_tagger_eng'
# ==========================================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) 

# Sample document
text = """Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language. 
Organizations today have large volumes of voice and text data from various communication channels like emails, text messages, social media newsfeeds, video, audio, and more. 
They use NLP software to automatically process this data, analyze the intent or sentiment in the message, and respond in real time to human communication."""

print("--- Original Text ---")
print(text[:150] + "...") # Print a snippet

# 1. Tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("\n--- Word Tokenization (First 10 words) ---")
print(words[:10])

# 2. Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if not w.lower() in stop_words and w.isalnum()]
print("\n--- After Stop Words Removal (First 10 words) ---")
print(filtered_words[:10])

# 3. POS Tagging
pos_tags = pos_tag(filtered_words)
print("\n--- POS Tagging (First 5 words) ---")
print(pos_tags[:5])

# 4. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]
print("\n--- Stemming (First 10 words) ---")
print(stemmed_words[:10])

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
print("\n--- Lemmatization (First 10 words) ---")
print(lemmatized_words[:10])

# 6. TF-IDF Calculation
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sentences)

print("\n--- TF-IDF Feature Names (First 10) ---")
print(vectorizer.get_feature_names_out()[:10])

print("\n--- TF-IDF Matrix Shape ---")
print(tfidf_matrix.shape)