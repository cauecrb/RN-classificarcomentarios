import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re

'''
rede neural utilizando tensorflow e keras para classificar comentarios como positivos ou negativos
a parte de tratamento de dados foi usada de acordo com a documentação do keras
'''

#começando a ler os dados das pastas
batch_size = 32

#aqui vamos pegars os dados negativos e positivos para treino e deixar 10% para validação
dados_treino = tf.keras.preprocessing.text_dataset_from_directory(
    "treino/train",
    batch_size=batch_size,
    validation_split=0.1,
    subset="training",
    seed=1217,
)
dados_validacao = tf.keras.preprocessing.text_dataset_from_directory(
    "treino/train",
    batch_size=batch_size,
    validation_split=0.1,
    subset="validation",
    seed=1217,
)

#aqui iremos pegar os dados que estao na pasta para testes quanto uma nova pasta usar o modulo de classificação
dados_teste = tf.keras.preprocessing.text_dataset_from_directory(
    "treino/test", batch_size=batch_size
)

dados_classificar = tf.keras.preprocessing.text_dataset_from_directory(
    "treino/classificar", batch_size=batch_size
)

#aqui iremos formatar o texto transformarem minusculas, trocar <br> por espaços.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

#constantes para informar o maximo de dados para o dicionario, usar lotes para comparção e padronizar
#o tamanho dos comentários
max_features = 10000
embedding_dim = 128
sequence_length = 512

#iremos transformar os comentarios em layers
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

#aqui iremos criar o vocabulario das palavras que mais se repetem
text_ds = dados_treino.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = layers.Embedding(max_features + 1, embedding_dim)(x)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# transformando os dados em vetores
train_ds = dados_treino.map(vectorize_text)
val_ds = dados_validacao.map(vectorize_text)
test_ds = dados_teste.map(vectorize_text)
classificar = dados_classificar.map(vectorize_text)

#Criando indices para o vocabulario
inputs = tf.keras.Input(shape=(None,), dtype="int64")

'''
Agora vamos criar a estrutura da rede com as funçoes de ativação
'''
#criar um modelo embedding que recebe os imputs
model = layers.Embedding(max_features, embedding_dim)(inputs)
#2 hidden layers com 16 perceptrons cada
model = layers.Dense(16, activation='relu')(model)
model = layers.Dense(16, activation='relu')(model)
#uma cama prediction
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(model)

#passando o modelo para novo objeto com os imputs e prediction
#este objeto  que sera usado para treino e validaçao
modelo = tf.keras.Model(inputs, predictions)

#usando o treinamento fit
modelo.fit(train_ds, validation_data=val_ds, epochs=epochs)

#validando o modelo com os dados de teste
modelo.evaluate(test_ds)