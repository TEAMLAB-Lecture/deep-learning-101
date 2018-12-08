# Deep Learning - Theory and Application(TEMLAB_DL2018)

## Instructor

## Reference Texts
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press., 2018 [link](https://github.com/janishar/mit-deep-learning-book-pdf)


## Syllabus and Lecture Resources
#### Week 1. Introduction to Deep Learning
- History of Deep Learning
- Programming environment for Deep Learning

#### Week x. Foundations of Deep Learning - Logistic Regression
#### Week x. Perceptron and Mulit Layer Perceptron
#### Week x. Building blocks of Deep Learning
#### Week x. Convolutional Neural Net
- Fundamentals of CNN
- Convolution & Pooling
- AlexNet
#### Week x. Performance Issues and Turning
- Data augumentation

#### Week x. Advanced CNN
#### Week x. Neural Embeddings
- Concepts of Distributed Word Representation
- Word2Vec
- Skip-gram & CBOW
- Negative Sampling
#### Week x. Recurrent Neural Net
- Fundamentals of RNN
- LSTM (Long Short-Term Memory model)
- GRU (Gated Recurrent Unit)
- Sequence2Sequence Model (Encoder-Decoder)
#### Week x. Learning Representation for unsupervised learning
- Autoencoder
- DEC
- ~~


#### Week x. Advanced Applications
- Neural Style
- Object Detection
  - RCNN, Fast(er)-RCNN
  - U-net
  - YOLO

#### Week x. Advanced Applications
  - Semantic Segmentation
  - FCN, DeconvNet, DeepLab
  - U-Net, Fusion Net, PSPNet

#### Week x. Genearive Models
- Variational Autoencoder
- Generative Adversarial Network
- Variants of GAN




<!-- ## Syllabus
#### ch 0. Programming environment setup
##### Python setup
  1. Python installation - [conda](https://www.youtube.com/watch?v=lqSNOIPGbns&index=5&list=PLBHVuYlKEkUJcXrgVu-bFx-One095BJ8I) , [atom](https://www.youtube.com/watch?v=cCxfLSIDfrk&index=6&list=PLBHVuYlKEkUJcXrgVu-bFx-One095BJ8I), [ML environment](https://www.youtube.com/watch?v=P4dOSb0jcUw&index=7&list=PLBHVuYlKEkUKnfbWvRCrwSuSeYh_QUlRl), [jupyter](https://www.youtube.com/watch?v=Hz_k_0sOv-w&index=8&list=PLBHVuYlKEkUKnfbWvRCrwSuSeYh_QUlRl)
  2. Pytorch - [Installation guide](./setup/README.md)
  3. Numpy - [Numpy in a nutshell](https://www.youtube.com/watch?v=aHthqCgsSFs&list=PLBHVuYlKEkULZLnKLzRq1CnNBOBlBTkqp)

##### Environments for deep learning machines
  - [Google Colab Tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)


#### ch 1. Introduction to NLP applications with Deep Learning
#### ch 2. Lanuage modeling
##### Class materials
| lecture | slide | video |
| --| --| --|
|A feature representation methof for text |[slide](https://1drv.ms/b/s!ApZ4mg7k2qYhgb9w0YVknfymIjTx4A) |~~video~~ |
|Languge Modeling |[slide](https://1drv.ms/b/s!ApZ4mg7k2qYhgb9w0YVknfymIjTx4A) | [video](https://vimeo.com/289888588)|
| Word embedding model - Word2vec |[slide](https://1drv.ms/b/s!ApZ4mg7k2qYhgb91RnMoYCOvh-Wg_g) |[video](https://vimeo.com/289888940) |
| Word2vec tricks - Hierarchical softmax & NCE loss |[slide](https://vimeo.com/292560864) |[video](https://vimeo.com/292559346) |
| GloVe & FastText |[slide](https://1drv.ms/b/s!ApZ4mg7k2qYhgcAfrAjTbMAYj7nsAQ) |[video](https://vimeo.com/292560272) |
| Sentence embeddings |[slide](https://1drv.ms/b/s!ApZ4mg7k2qYhgcAhuHt-u821RngETQ) |[video](https://vimeo.com/292560864) |

##### Reference papers
- [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [word2vec_explained](https://arxiv.org/pdf/1402.3722.pdf)
- [doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
- [fasttext](http://aclweb.org/anthology/Q/Q17/Q17-1010.pdf)
- [t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
- [Evaluation methods for unsupervised word embeddings](http://aclweb.org/anthology/D15-1036)

##### Dataset
- [A Million News Headlines](https://www.kaggle.com/therohk/million-headlines/)

##### Reading Materials - papers
| Name | URL | slide | video |
| ---  | ---- | ----| --- |
| Graph2Vec | https://link.springer.com/chapter/10.1007/978-3-319-73198-8_9 |  |  |
| Entity2Vec |http://www.di.unipi.it/~ottavian/files/wsdm15_fel.pdf|  |  |
| WordNet2Vec| https://arxiv.org/abs/1606.03335|  |  |
|Author2Vec |https://www.microsoft.com/en-us/research/publication/author2vec-learning-author-representations-by-combining-content-and-link-information/ |  [slide](slide/author2vec.pdf) |[video](https://vimeo.com/290894287)  |
|Paper2Vec |https://arxiv.org/pdf/1703.06587.pdf |  |  |
|Wikipedia2Vec |[github](https://wikipedia2vec.github.io/wikipedia2vec/), [paper](http://www.aclweb.org/anthology/K16-1025) | [slide](slide/wikipedia2vec.pdf) | [video](https://vimeo.com/290916448) |
|Sense2Vec |https://arxiv.org/abs/1511.06388 | [slide](sldie/sense2vec.pdf) |  [video](https://vimeo.com/290891986) |
|Ngram2Vec |http://www.aclweb.org/anthology/D17-1023 |  |  |
|morphology embeddings |http://aclweb.org/anthology/W/W13/W13-3512.pdf |  |  |
|char embeddings |http://aclweb.org/anthology/D15-1176 | [slide](https://docs.google.com/presentation/d/12QsX5wI3JwDkSq5pROP-v2-0JQutGLwuMSPPJKkv_Fk/edit?usp=sharing), |  [video](https://vimeo.com/290892980/e0a8501abc) |


##### Reading Materials - Blog
- [빈도수 세기의 놀라운 마법](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/)
- [Word embeddings: exploration, explanation, and exploitation](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
- Word2Vec overall
  - [word2vec tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
  - [QA: Word2Vec Actual Target Probability from TensorFlowKR](https://www.facebook.com/groups/TensorFlowKR/permalink/743666392641088/)
  - [On word embeddings - Part 1](http://ruder.io/word-embeddings-1/)
- Hierarchical Softmax & Negative Sampling
  - [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
  - [Hierarchical Softmax](http://dalpo0814.tistory.com/7)
  - [Hierarchical Softmax](http://building-babylon.net/2017/08/01/hierarchical-softmax/)
  - [Hugo Larochelle's Lecture - hierarchical output layer](https://www.youtube.com/watch?v=B95LTf2rVWM)
  - [On word embeddings - Part 2: Approximating the Softmax](http://ruder.io/word-embeddings-softmax/)
- Visualization
  - [PCA vs t-SNE](https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b)
  - [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- Trends of Word Embeddings
  - [Awesome2Vec](https://github.com/MaxwellRebo/awesome-2vec)
  - [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)


#### ch 3. Neural Network Archietures for NLP tasks

##### Class materials
| lecture | slide | video |
| --| --| --|
| Convolutional Neural Network |[slide]() | ~~video~~ |
| Text classification task  | [slide]()  | ~~video~~  |
| CNN for Text Classification (words) |[slide]() | ~~video~~ |
| CNN for Text Classification (characters) |[slide]() | ~~video~~ |
| Recurent Neural Networks |[slide]() | ~~video~~ |
| RNN for text tasks |[slide]() | ~~video~~ |
| Datasets and Tricks |[slide]() | ~~video~~ |


##### Reference papers
- [LSTM](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/43905.pdf)
- [GRU](https://arxiv.org/pdf/1412.3555.pdf)
- [Convolutional Neural Networks for Sentence Classification](http://www.people.fas.harvard.edu/~yoonkim/data/sent-cnn.pdf)
- [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
- [Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts](http://www.aclweb.org/anthology/C14-1008)
- [Dimensional Sentiment Analysis Using a Regional CNN-LSTM Model](http://anthology.aclweb.org/P16-2037)
- [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://www.aclweb.org/anthology/D13-1170)


##### Reading Materials - Blog
- LSTM Networks for Sentiment Analysis¶: http://deeplearning.net/tutorial/lstm.html
- ext By the Bay 2015: https://www.youtube.com/watch?v=tdLmf8t4oqM
- How to solve 90% of NLP problems: a step-by-step guide
 https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e

##### Reading Materials - papers
| Name | URL | slide | video |
| ---  | ---- | ----| --- |
| BB_twtr at SemEval-2017 Task 4 | https://arxiv.org/abs/1704.06125 |  |  |
| Sentiment Classification  |http://www.aclweb.org/anthology/D15-1167   |   |   |
|Fast2Text Classification   |  https://arxiv.org/pdf/1607.01759.pdf |   |   |
|Automated Essay Scoring   |http://www.aclweb.org/old_anthology/D/D16/D16-1193.pdf   |   |   |
| Grammatical Error Correction |https://arxiv.org/abs/1801.08831   |   |   |
|Character Language Models   | https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017|   |   |
|NER  |https://arxiv.org/pdf/1603.01360.pdf|   |   |
|CNN for Modelling Sentences   | http://aclweb.org/anthology/P/P14/P14-1062.pdf  |   |   |


#### ch 4. Machine Translation and Attention Mechanism
##### Class materials
| lecture | slide | video |
| --| --| --|
| Introduction to MT |[slide]() | ~~video~~ |
| Attnetion Mechanism  | [slide]()  | ~~video~~  |
| Transformer | [slide]() | [video](https://vimeo.com/303863180)  |

##### Reference papers
- [S2S](https://arxiv.org/pdf/1409.3215.pdf)
- [attention_paper](https://arxiv.org/abs/1409.0473)
- [BLUE](https://www.aclweb.org/anthology/P02-1040.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
-
##### Reading Materials - Blog
- [Sequence-to-Sequence 모델로 뉴스 제목 추출하기](https://ratsgo.github.io/natural%20language%20processing/2017/03/12/s2s/)
- [Attention 매카니즘](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/)
-
- [Attention API로 간단히 어텐션 사용하기](http://freesearch.pe.kr/archives/4876)
- [Recursive Neural Networks](https://ratsgo.github.io/deep%20learning/2017/04/03/recursive/)


##### Reading Materials - papers
- [Pervasive Attention](https://arxiv.org/abs/1808.03867.pdf)
- https://arxiv.org/abs/1810.00660
- [BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies](https://arxiv.org/abs/1511.06909)
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)


#### ch 5. Question and Answering Models

##### Reading Materials - papers
- [
Personalizing Dialogue Agents](https://arxiv.org/abs/1801.07243), [dataset](http://parl.ai/)

##### Reading Materials - Blog
- [Chat Smarter with Allo](https://ai.googleblog.com/2016/05/chat-smarter-with-allo.html) -->


<!-- #### ch 6. Dependency Parsing

##### Reading Materials - Blog
https://medium.com/@anupamme/paper-reading-1-assessing-the-ability-of-lstms-to-learn-syntax-sensitive-dependencies-by-linzen-739cec9d0212
https://ratsgo.github.io/korean%20linguistics/2017/04/29/parsing/


## Assignments

## Final Project



### Reference
https://github.com/keon/awesome-nlp/blob/master/README.md
https://github.com/dparlevliet/awesome-nlp

https://arxiv.org/pdf/1412.3555.pdf
https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/43905.pdf
https://arxiv.org/pdf/1712.00170.pdf
https://arxiv.org/abs/1406.2661 -->


## References
https://www.youtube.com/watch?v=IKHzc7sGCxQ&feature=youtu.be&fbclid=IwAR3guScQRGjFOWuthjJbrrWP21K0mOt-fZTkFnIy95eQLhfbTHwZ0lxxPy8
