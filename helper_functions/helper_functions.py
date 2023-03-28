import pandas as pd
import zipfile
import tensorflow as tf

def remove_stopwords(sentence):
    """
    Removes a list of stopwords
    
    Args:
        sentence (string): sentence to remove the stopwords from
    
    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
             "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
             "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
             "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most",
             "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
             "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", 
             "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
             "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", 
             "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd",
             "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", 
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", 
             "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    
    # Sentence converted to lowercase-only
    sentence = sentence.lower()
    sentence_splitted = sentence.split(" ")
    sentence_removed = []

    for word in sentence_splitted:
        # print(f"current word {word}")
        if word not in stopwords:
          sentence_removed.append(word)

    return ' '.join(sentence_removed)


def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a zipped CSV file
    
    Args:
        filename (string): path to the CSV file
    
    Returns:
        sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """
    # Specify the name of the zip file to unzip
    zip_file = zipfile.ZipFile("C:\\Projects\\Project1\\Twitter-NLP\\data\\dataset\\archive.zip")

    # Specify the directory to extract the files to
    extract_dir = "C:\\Projects\\Project1\\Twitter-NLP\\data\\dataset\\"

    # Extract all the files from the zip file to the specified directory
    zip_file.extractall(extract_dir)

    # Close the zip file
    zip_file.close()

    # Read the csv as a pandas df
    df = pd.read_csv(filename, 
                    delimiter=",", 
                    encoding="latin",
                    names=["label", "id", "time", "query", "user", "tweet"])
    
    labels = df["label"] 
    sentences = df["tweet"]
    filtered_sentences = []
    for sentence in sentences:
        filtered_sentences.append(remove_stopwords(sentence))

    return filtered_sentences, labels


def fit_tokenizer(train_sentences, num_words, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences
    
    Args:
        train_sentences (list of string): lower-cased sentences without stopwords to be used for training
        num_words (int) - number of words to keep when tokenizing
        oov_token (string) - symbol for the out-of-vocabulary token
    
    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
        
    # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words, oov_token=oov_token)
    # Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)
        
    return tokenizer


def seq_and_pad(sentences, tokenizer, padding, maxlen):
    """
    Generates an array of token sequences and pads them to the same length
    
    Args:
        sentences (list of string): list of sentences to tokenize and pad
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        padding (string): type of padding to use
        maxlen (int): maximum length of the token sequence
    
    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """    
    ### START CODE HERE
       
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the correct padding and maxlen
    padded_sequences = tf.keras.utils.pad_sequences(sequences,
                                                    maxlen=maxlen,
                                                    padding=padding)    
    ### END CODE HERE
    
    return padded_sequences


def train_val_split(sentences, labels, training_split):
    """
    Splits the dataset into training and validation sets
    
    Args:
        sentences (list of string): lower-cased sentences without stopwords
        labels (list of string): list of labels
        training split (float): proportion of the dataset to convert to include in the train set
    
    Returns:
        train_sentences, validation_sentences, train_labels, validation_labels - lists containing the data splits
    """
    
    
    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(len(sentences) * training_split)

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]
        
    return train_sentences, validation_sentences, train_labels, validation_labels