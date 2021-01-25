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
