# Rede Neural para classificar comentarios, como positivos ou negativos, de Filmes

Rede neural desenvolvida com TensorFlow e Keras, para classificar comentario como positivos ou negativos utilizando a base Imdb. 


## bibliotecas

keras e TensorFlow

para instalar a api e exutar, use o seguinte comando:
```bash
pip install keras
```
```bash
pip instal tensorflow
```


### colocando os arquivos para treino e teste

baixe os arquivos de treino e teste em  ​http://ai.stanford.edu/~amaas/data/sentiment/​
o local que os arquivos devem ser colocados respeita a estrutura de pastas do zip, dentro da pasta treino.
ex: treino/train/pos
 para arquivos de treino positivos

a rede tambem possui um modulo para validação que utiliza a pasta classi, neste diretorio devem ser colocados os comentarios q serao classificados apos a rede neural ser treinada.



## utilizando

após instalar as dependências e colocar os arquivos nos diretorios correspondentes, rode o arquivo RdeNeuralClassificacao.py.
ele ira mostrar a convergencia e os erros apos o treino, tambem gerará um arquivo model.h5, que contem o modelo da rede.
Para  poder efetuar a classificaçao de um comentario, a rede deve ter aprendido previamente, o arquivo desejado tem que ser colocado em:
treino/classificar/classi
Após colocar o arquivo que contém o comentario desejado, deve-se rodar o trecho de codigo indicado no código.

## Sobre autor
Cauê Rafael Burgardt crburgardt@gmail.com. =D

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Neural network to classify film reviews as positive or negative

Neural network developed with TensorFlow and Keras, to classify comments as positive or negative using the Imdb base.


## libraries

keras and TensorFlow

to install the api and exutar, use the following command:
`` bash
pip install keras
``
`` bash
pip instal tensorflow
``


### putting the files for training and testing

download the training and test files at http://ai.stanford.edu/~amaas/data/sentiment/
the location where the files must be placed respects the zip folder structure, inside the training folder.
ex: training / train / pos
 for positive training files

the network also has a module for validation that uses the classi folder, in this directory comments must be placed q will be classified after the neural network is trained.



## using

after installing the dependencies and placing the files in the corresponding directories, run the file RdeNeuralClassificacao.py.
it will show convergence and errors after training, it will also generate a model.h5 file, which contains the network model.
To be able to classify a comment, the network must have previously learned, the desired file must be placed in:
training / classifying / classi
After placing the file that contains the desired comment, you must run the code section indicated in the code.

## About author
Cauê Rafael Burgardt crburgardt@gmail.com. = D
