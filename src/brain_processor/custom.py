import numpy as np

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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


def custom_nlu_embed_model():
    # init tokenizer
    tk = Tokenizer()
    tk.fit_on_texts(utterances)

    # get size of vocabulary for embeding layer
    vocab_size = len(tk.word_index) + 1

    encoded_docs = tk.texts_to_sequences(utterances)
    print(encoded_docs)

    


if __name__ == "__main__":
    custom_nlu_embed_model()

