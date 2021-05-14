import math
from collections import defaultdict, OrderedDict
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


def cosine_similarity(vec1, vec2):
    """
    Cosine similarity function between vec1 and vec2
    :param vec1: dictionary vector 1
    :param vec2: dictionary vector 2
    :return: cosine similarity
    """
    # we assume these come in as dictionaries because that's how dist freq is usually kept
    vec1 = [v for v in vec1.values()]
    vec2 = [v for v in vec2.values()]
    dot_product = 0
    for i, num in enumerate(vec1):
        dot_product += vec2[i] * num
    # len is for euclidean distance
    len_vec1 = math.sqrt(sum([n**2 for n in vec1]))
    len_vec2 = math.sqrt(sum([n**2 for n in vec2]))

    return dot_product / (1 + (len_vec1 * len_vec2))


def inverse_doc_freq(docs, word):
    """
    Calculates IDF
    :param docs: list of documents
    :param word: term
    :return: IDF
    """
    frequency = 0
    for document in docs:
        if word in document:
            frequency += 1
    return len(docs) / (1 + frequency)


def term_freq(doc, word):
    """
    Calculates TF
    :param doc: list of documents
    :param word: term
    :return: TF
    """
    count = 0
    for w in doc:
        if w.lower() == word.lower():
            count += 1
    return count / len(doc)


def get_freq_dist(doc):
    dist = defaultdict(lambda: 0)
    for word in doc:
        dist[word] += 1
    size = len(doc)
    for (word, count) in dist:
        dist[word] = count / size
    return dist


def get_word_counts(doc):
    dist = defaultdict(lambda: 0)
    for word in doc:
        dist[word] += 1
    return dist


def td_idf_erizer(sentence, lexicon, idfs):
    vec = OrderedDict((token, 0) for token in lexicon)
    word_counts = get_word_counts(tokenizer.tokenize(sentence))
    for word, count in word_counts.items():
        if word not in lexicon:
            continue
        tf = count / len(word_counts)
        vec[word] = (math.log2(tf) * math.log2(idfs[word]))
    return vec


def tf_idf(docs):
    lexicon = set()
    for doc in docs:
        lexicon.update(tokenizer.tokenize(doc))
    idfs = {}
    i = 0
    for word in lexicon:
        idfs[word] = inverse_doc_freq(docs, word)
        i += 1

    result = []
    tokenized_docs = [tokenizer.tokenize(doc) for doc in docs]

    for doc in tokenized_docs:
        vec = OrderedDict((token, 0) for token in lexicon)
        word_counts = get_word_counts(doc)
        for word, count in word_counts.items():
            tf = count / len(word_counts)
            vec[word] = (math.log2(tf) * math.log2(idfs[word]))
        result.append(vec)

    return [result, lexicon, idfs]




