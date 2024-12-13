import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import string

# Завантаження необхідних ресурсів NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def process_book(text):
    first = text.find('CHAPTER I.')
    start_index = text.find('CHAPTER I.', first + 1)
    main_text = text[start_index:]
    main_text = re.sub(r'[\“\”\'\’]', '', main_text)
    main_text = re.sub('[^a-zA-Z]', ' ', main_text)
    return main_text

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]
    return tokens

def create_corpus_dict(chapters):
    texts = [preprocess_text(chapter) for chapter in chapters]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary

def run_lda(corpus, dictionary, num_topics=12):
    lda_model = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    return lda_model

def process_chapters(text):
    chapters = text.split('CHAPTER')[1:]
    tfidf_vectorizer = TfidfVectorizer()
    chapter_top_words = []
    for chapter in chapters:
        processed_tokens = preprocess_text(chapter)
        processed_text = ' '.join(processed_tokens)
        tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().flatten()
        sorted_indices = tfidf_scores.argsort()[::-1]
        top_words = [(feature_names[idx], tfidf_scores[idx]) for idx in sorted_indices[:20]]
        chapter_top_words.append(top_words)
    return chapter_top_words, chapters

def main():
    with open('Alices_Adventures_in_Wonderland.txt', 'r', encoding='utf-8') as file:
        book = file.read()
    text = process_book(book)
    chapter_top_words, chapters = process_chapters(text)
    corpus, dictionary = create_corpus_dict(chapters)
    lda_model = run_lda(corpus, dictionary)
    print("\nTF-IDF Results:")
    for idx, top_words in enumerate(chapter_top_words, 1):
        print(f"Топ-20 слів для глави {idx}: {top_words}\n")

    print("\nLDA Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Тема {idx + 1}: {topic}\n")

if __name__ == '__main__':
    main()
