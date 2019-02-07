import numpy as np

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Sequential, layers

# x_train
utterances = ['Well done!',
              'Good work',
              'Great effort',
              'nice work',
              'Excellent!',
              'Weak',
              'Poor effort!',
              'not good',
              'poor work',
              'Could have done better.'
              ]
# y_train
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

GLOVE_PATH = '../static/glove/glove.6B.100d.txt'


def custom_nlu_embed_model():
    """ Used GLOVE vectors in Embedding layer - trainable=False
    possible variations:
    a. Use embedding layer as in example below
    b. prepare training set already vectorised from GLOVE data and fit to model (in such case you do not have to
    create new mdel for new incomed words)
    """
    # init tokenizer
    tk = Tokenizer()
    tk.fit_on_texts(utterances)

    # get size of vocabulary for embeding layer
    vocab_size = len(tk.word_index) + 1

    encoded_docs = tk.texts_to_sequences(utterances)
    print(encoded_docs)

    max_length = 4

    # make padding
    padded_utterances = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_utterances)

    # load data from glove dict
    glove_index = dict()
    file = open(GLOVE_PATH)

    for line in file:
        splitted_line = line.split()
        word = splitted_line[0]
        coeficient = np.asarray(splitted_line[1:], dtype='float32')

        glove_index[word] = coeficient
    file.close()

    print(f'Loaded {len(glove_index)} vectors ============>>>>>>')
    print('good* vector: ', glove_index['good'])

    GLOVE_VECTOR_SIZE = 100

    # weight matrix for embedding layer
    embed_matrix = np.zeros((vocab_size, GLOVE_VECTOR_SIZE))

    for word, i in tk.word_index.items():
        em_vector = glove_index.get(word)

        if em_vector is not None:
            embed_matrix[i] = em_vector

    # define model
    model = Sequential()

    em_layer = layers.Embedding(vocab_size, GLOVE_VECTOR_SIZE, weights=[embed_matrix], input_length=max_length, trainable=False)

    model.add(em_layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    # get word from vector
    # layer_emb = em_layer.get_weights()[0]
    #
    # words_embeddings = {w:layer_emb[idx] for w, idx in tk.word_index.items()}
    # print('embeddings:', words_embeddings)

    model.fit(padded_utterances, labels, epochs=50, verbose=0)



if __name__ == "__main__":
    custom_nlu_embed_model()

