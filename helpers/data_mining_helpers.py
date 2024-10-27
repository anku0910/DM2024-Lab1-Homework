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

@staticmethod
    def format_rows(docs):
        """Format text and strip special characters"""
        return [[" ".join(d.split("\n")).strip()] for d in docs.data]
    
    @staticmethod
    def format_labels(target, docs):
        """Format document labels"""
        return docs.target_names[target]
    
    @staticmethod
    def check_missing_values(row):
        """Count missing values in dataframe"""
        return ("The amount of missing records is: ", sum(row))
    
    @staticmethod
    def tokenize_text(text, remove_stopwords=False):
        """Tokenize English text"""
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            tokens.extend(nltk.word_tokenize(sent, language='english'))
        return tokens
    
    @staticmethod
    def get_similarity_matrix(tdm_df, categories):
        """Calculate document similarity matrix"""
        df = tdm_df.copy()
        df['cat'] = categories
        df.sort_values('cat', inplace=True)
        
        idx = [f"{i}-{df.loc[i, 'cat']}" for i in df.index]
        df = df.drop('cat', axis=1)
        
        return pd.DataFrame(
            cosine_similarity(df.values),
            columns=idx,
            index=idx
        )