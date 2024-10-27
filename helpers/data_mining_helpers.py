from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk

"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens
    
########################################
""" My Data Mining Helper Functions """
########################################
# Function to create term-document matrix DataFrame for each category
# (by default, CountVectorizer is used to create term-document frequent matrix)
def create_term_document_df(text, vectorizer=CountVectorizer(), index=None):
    X = vectorizer.fit_transform(text)  # Transform the text data into word counts, which is a sparse matrix representation
    
    # Get the unique words (vocabulary) from the vectorizer
    words = vectorizer.get_feature_names_out()
    
    # Create a DataFrame where rows are documents and columns are words
    term_document_df = pd.DataFrame(X.toarray(), columns=words, index=index)
    
    return term_document_df

# Filter the bottom 1% and top 5% words based on their sum across all documents
def filter_top_bottom_words_by_sum(tdm_df, top_percent=0.05, bottom_percent=0.01, verbose=False):
    # Calculate the sum of each word across all documents
    word_sums = tdm_df.sum(axis=0)
    
    # Sort the words by their total sum
    sorted_words = word_sums.sort_values()
    
    # Calculate the number of words to remove
    total_words = len(sorted_words)
    top_n = int(top_percent * total_words)
    bottom_n = int(bottom_percent * total_words)
    
    # Get the words to remove from the top 5% and bottom 1%
    words_to_remove = pd.concat([sorted_words.head(bottom_n), sorted_words.tail(top_n)]).index

    if verbose:
        print(f'Bottom {bottom_percent*100}% words: \n{sorted_words.head(bottom_n)}') #Here we print which words correspond to the bottom percentage we filter
        print(f'Top {top_percent*100}% words: \n{sorted_words.tail(top_n)}') #Here we print which words correspond to the top percentage we filter
        
    # Return (1) the DataFrame without the filtered words and (2) words removed
    return tdm_df.drop(columns=words_to_remove), words_to_remove
