import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tensorflow import keras
import numpy as np

train_data = [    "Eu amo o meu cachorro, ele é muito fofo",    "Esse filme é emocionante e me fez chorar",    "O trânsito está muito congestionado hoje",    "Eu fui para a praia e peguei um sol maravilhoso",    "Essa música é muito animada e me faz querer dançar",    "Meu time de futebol perdeu o jogo de ontem",    "Essa comida está muito ruim, não gostei",    "Eu me diverti muito na festa de aniversário",    "A prova de matemática estava muito difícil",    "Essa cidade é muito bonita, gostei bastante"]

train_labels = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1])

# Pré-processamento dos dados de texto
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_data)
data = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Definição do modelo
model = keras.Sequential([
    keras.layers.Embedding(len(word_index)+1, 16, input_length=20),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinamento do modelo
model.fit(data, train_labels, epochs=50)

def predict_sentiment(model, tokenizer, text):
    # encode text using tokenizer
    encoded_text = tokenizer.texts_to_sequences([text])
    encoded_text = pad_sequences(encoded_text, maxlen=20, padding='post', truncating='post')

    # make prediction
    prediction = model.predict(encoded_text)[0][0]

    # convert prediction to human-readable sentiment
    if prediction < 0.5:
        sentiment = 'Negative'
    else:
        sentiment = 'Positive'

    return sentiment

text = 'adorei'
sentiment = predict_sentiment(model, tokenizer, text)
print(sentiment)
