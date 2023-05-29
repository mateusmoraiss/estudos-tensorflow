import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tensorflow import keras
import numpy as np

train_data1 = [        "Eu amo o meu cachorro, ele é muito fofo",        "Esse filme é emocionante e me fez chorar",        "O trânsito está muito congestionado hoje",        "Eu fui para a praia e peguei um sol maravilhoso",        "Essa música é muito animada e me faz querer dançar",        "Meu time de futebol perdeu o jogo de ontem",        "Essa comida está muito ruim, não gostei",        "Eu me diverti muito na festa de aniversário",        "A prova de matemática estava muito difícil",        "Essa cidade é muito bonita, gostei bastante",    "Eu adoro viajar e conhecer novos lugares",    "O novo álbum do meu artista favorito é incrível",    "Eu amo passar tempo com meus amigos e família",    "Não gosto de acordar cedo para trabalhar",    "Eu me sinto renovado depois de praticar exercícios físicos",    "Acho que esse restaurante tem a melhor comida da cidade",    "Não gosto de assistir filmes de terror, me dão medo",    "Adoro experimentar novas comidas e receitas",    "Acho que a primavera é a estação mais bonita do ano",    "Não gosto de dirigir em dias de chuva"]

train_labels1 = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
train_data2 = [    "Eu adoro ir ao cinema assistir filmes de comédia",    "Acredito que a educação é a base para uma sociedade melhor",    "Não gosto de comida apimentada",    "Adoro passar tempo lendo livros",    "O tempo está bastante instável hoje, com chuva e sol intercalados",    "Acho que a música é capaz de nos transportar para outros lugares",    "Não sou muito fã de redes sociais",    "Fiz uma viagem incrível para a Europa no ano passado",    "Acho que a tecnologia é muito importante para o nosso dia a dia",    "Não gosto de lugares muito cheios e agitados",    "Adoro tomar um bom café",    "Acho que a arte é uma forma de expressão muito poderosa",    "Não suporto injustiças",    "Gosto muito de animais e tenho um gato em casa",    "Adoro cozinhar e experimentar novas receitas",    "Acredito que a gentileza é uma virtude importante",    "Não gosto de falar em público",    "Adoro passar tempo ao ar livre em contato com a natureza",    "Acho que o amor é um sentimento muito bonito",    "Não gosto de me sentir preso em lugares pequenos",    "Gosto de assistir a jogos de futebol no estádio",    "Adoro fazer compras em lojas de departamento",    "Acho que a meditação pode ser muito benéfica para a saúde mental",    "Não gosto de pessoas que são desonestas",    "Adoro praticar esportes radicais",    "Acredito que a música clássica é uma forma de arte muito sofisticada",    "Não suporto intolerância e preconceito",    "Adoro viajar para lugares exóticos",    "Acho que é importante cuidar do meio ambiente",    "Gosto de ficar em casa e assistir a séries de TV"]

train_labels2 = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])



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

text = 'odiei'
sentiment = predict_sentiment(model, tokenizer, text)
print(sentiment)
