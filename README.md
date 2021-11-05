# Neurovisual Controller Celeste

[English](./README-en.md)

Este projeto consistiu em explorar a aplicação de um controlador neurovisual 
ao ambiente do jogo digital Celeste. 

Durante o desenvolvimento, os quadros do jogo foram capturados, encolhidos e posteriormente desfocados. 
Em seguida, foram dispostos no procedimento de *K-Means clustering* para quantização de cores. 
Além disso, foram dispostos algoritmos de *Background Subtraction* de tipo KNN, MOG2, CNT, GMG, GSOC, LSBP e MOG, para ajudarem no 
rastreamento da personagem, mas a oscilação da câmera do jogo nos cenários do capítulo Prólogo impediram o processo. O problema de rastreamento foi resolvido com o algoritmo de rastreamento de objetos do módulo OpenCV CSRT, 
e mostrou-se promissor tanto no primeiro quanto no segundo cenário do Prólogo. 

Infelizmente este estudo não conseguiu partir para as etapas de aplicação da Inteligência Artificial no jogo, 
devido ao esforço do projeto ter sido voltado completamento para o Processamento Digital de Imagem dos quadros do jogo.

Para mais informações e trabalhos futuros, veja o artigo em: **LINK**  
O conjunto de imagens referentes aos quadros do jogo, utilizados neste estudo podem ser obtidas em: [AQUI](https://drive.google.com/drive/folders/1YwSanqiYwS9-Y56eva9azxy7BfaU2H64?usp=sharing)

# Pacotes

```shell
pip install opencv-python
pip install tensorflow
```

### Pacotes opcionais

Para obter o nome da janela do aplicativo de destino
```shell
pip install PyGetWindow
```

Para ter mais métodos de subtração de fundo (opencv.bgsegm) 
```shell
pip install opencv-contrib-python
```

Para realizar testes de registro de imagem 
```shell
pip install scikit-image
```