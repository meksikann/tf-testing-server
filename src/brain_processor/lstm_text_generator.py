from os.path import join, dirname
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from pickle import dump, load
from tensorflow.python.keras.preprocessing import sequence
tf.enable_eager_execution()


def load_txt_doc(filename):
    """get row text from file"""
    file = open(filename, 'r')
    text = file.read()

    file.close()
    return text


def clear_text(text):
    text = text.replace('--', ' ')

    # split text into tokens
    text_tokens = text.split()

    table = str.maketrans('', '', string.punctuation)
    text_tokens = [tk.translate(table) for tk in text_tokens]

    text_tokens = [word.lower() for word in text_tokens if word.isalpha()]

    return text_tokens


def generate_token_sequences(tokens):
    length = 51
    sequences = list()

    for i in range(length, len(tokens)):
        seq = tokens[i-length: i]

        line = ' '.join(seq)

        sequences.append(line)

    return sequences


def save_sequences(array, finename):
    data = '\n'.join(array)
    file = open(finename, 'w')
    file.write(data)
    file.close()


def create_model(net_units, vocab_size, sequence_length):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 50, input_length=sequence_length))
    model.add(keras.layers.LSTM(net_units, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(net_units))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(net_units, activation='relu'))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))

    return model


def start_training():
    file_name = 'republic.txt'
    seq_filename = 'republic_seq.txt'
    neurons_number = 100
    batch_size = 120

    text = load_txt_doc(join(dirname(__file__), file_name))

    clean_tokens = clear_text(text)
    sequences = generate_token_sequences(clean_tokens)

    # save to file
    # save_sequences(sequences, join(dirname(__file__), seq_filename))

    # tokenize sequences using tf tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    tok_sequences = tokenizer.texts_to_sequences(sequences)

    vocab_size = len(tokenizer.word_index) + 1

    # keras.utils.to_categorical
    # generate X and Y, make Y-onehot encoding

    tok_sequences = np.array(tok_sequences)
    x_set, y_set = tok_sequences[:, :-1], tok_sequences[:, -1]

    y_set = keras.utils.to_categorical(y_set, num_classes=vocab_size)

    embed_length = x_set.shape[1]

    model = create_model(neurons_number, vocab_size, embed_length)
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(x_set, y_set, batch_size=batch_size, epochs=5)

    # save model weights and tokenizer data
    model.save( join(dirname(__file__), "weights/tf_lstm.h5"))
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
    print('================>>>>>>>>>>>>>>>>DONE<<<<<<<<<<<<<<<<<=============')


def get_word(vocab, idx):
    res = ''
    for word, i in vocab:
        if i == idx:
            res = word
            break

    return res


def generate_string(tk, model, seq_length, input_string, output_length):
    res = list()
    x_text = input_string

    for i in range(output_length):
        tokenized_seq = tk.texts_to_sequences([x_text])[0]
        tokenized_seq = sequence.pad_sequences([tokenized_seq], maxlen=seq_length, truncating='pre')
        prediction = model.predict_classes(tokenized_seq, verbose=0)

        # word = get_word(tk.word_index.items(), prediction[0])

        word_f = ''

        for word, i in tk.word_index.items():
            if i == prediction[0]:
                word_f = word
                break
        x_text += ' ' + word_f
        res.append(word_f)

    return ' '.join(res)


def start_prediction(input_string):
    filename = 'republic_seq.txt'
    output_length = 10000

    doc = load_txt_doc(join(dirname(__file__), filename))
    lines = doc.split('\n')
    length = len(lines[0].split()) - 1

    model = keras.models.load_model(join(dirname(__file__), 'weights/tf_lstm.h5'))

    tokenizer = load(open('tokenizer.pkl', 'rb'))

    x_test = generate_string(tokenizer, model, length, input_string, output_length)
    print('**************** ', x_test)



# start_training()
start_prediction('his name and got it')
