# Deep Learning - Theory and Application

## 강의 정보
* 강좌명: Deep Learning - Theory and Application
* 강의자명: 가천대학교 산업경영공학과 최성철 교수 (sc82.choi@gachon.ac.kr, Director of [TEAMLAB](http://theteamlab.io/))
* Email: teamlab.gachon@gmail.com

## Reference Textbooks
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press., 2018 [link](https://github.com/janishar/mit-deep-learning-book-pdf)

## Reference Lectures
- 모두를 위한 머신러닝/딥러닝 강의 (김성훈, 2017) - [link](http://hunkim.github.io/ml/?fbclid=IwAR3R-w_qdbZgfr9HFm7b_thvxnKtYuAjqhMV2UlRFXQl7iXffVdhMASOd1k)
- Deep Learning (deeplearning.ai, 2018) - [link]()
- https://www.youtube.com/watch?v=SGZ6BttHMPw&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH

## Syllabus and Lecture Resources
## 강의 구성

### Chapter 1: Introduction to Deep Learning
#### Lecture
- 왜 딥러닝의 시대가 시작되었는가?
- 딥 러닝의 역사
- 딥 러닝 Applications
- 딥 러닝 Framework 비교
- 딥 러닝 Framework 설치
  - Tensorflow
  - Keras

#### Reading materials
- [History and Background](https://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html)
- [윈도우에 Keras 설치하기](https://tykimos.github.io/2017/08/07/Keras_Install_on_Windows/)

##### Articles
- Deep Learning Hardware Guide - [link](http://timdettmers.com/2018/12/16/deep-learning-hardware-guide/?fbclid=IwAR00w6IMdsIzAgeogyD7m87_qnoHxgk_FKCrZ5sLBRFfu3K2f6P4JNI6Rv0)

### Chapter 2: Foundations of Deep Learning - Logistic Regression
#### Lecture
- Deep Learning Ecosystem
- TensorFlow Overview

#### Coding environment
- [Google Drive Colaboratory ], todaycodes오늘코드
(https://www.youtube.com/watch?v=XRBXMohjQos&list=PLaTc2c6yEwmo9MZi-0OLi8F6bM6AA0wjE)
- [Colaboratory](https://colab.research.google.com)
- [GPU on AWS](https://beomi.github.io/2018/03/18/Create_GPU_spot_EC2_for_ML/)
- [AWS Deep Learning AMI](https://aws.amazon.com/ko/machine-learning/amis/)


### Chapter 3: Perceptron and Multi Layer Perceptron
#### Lecture
- Multi Layer Perceptron Overview - [강의영상](https://vimeo.com/327559376)
- Computation Graph - [강의영상](https://vimeo.com/327559472)
- Backpropagtaion - [강의영상](https://vimeo.com/327559448)
- Activiation functions - [강의영상](https://vimeo.com/327559432)
- Hidden Units
- Build MLP Model with TF
- Build MLP Model with Keras

#### Reading materials
- Backpropagation 설명자료 - [link](https://www.facebook.com/groups/TensorFlowKR/permalink/432969130377484/)

### Chapter 4: Network Turning
#### Lecture
- Network Turning Overview - [강의영상](https://vimeo.com/329287153/286e5cee0f)
- Data augumentation - [강의영상](https://vimeo.com/329287180/71893cb2e1)
- Mini-batch SGD - [강의영상](https://vimeo.com/329287123/c230292555)
- Early stopping
- l2 Regularization
- Dropout
- Weight initialization
- Learnin rate decay
- Optimizer Overview
- Momemntum Optimizer
- Adaptive Optimizer
- Batch Normalization
- Hyper Parameter Turning

#### Reading materials
- [자습해도 모르겠던 딥러닝, 머리속에 인스톨 시켜드립니다.](https://www.slideshare.net/yongho/ss-79607172), 하용호, 2017
- [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039), 2017
- [Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)
- [Batch Normalization (ICML 2015)](http://sanghyukchun.github.io/88/)
- An Excessively Deep Dive Into Natural Gradient Optimization - [link](https://databreak.netlify.com/2019-04-13-Natural_Gradient_Optimzation/?fbclid=IwAR0JxxbStBUTaNzI5bKuxa8PqO9b3pxpM3kn5AKj5gMVtUysAJo57w-MeYo)
- Adam Optimization Algorithm - [link](https://engmrk.com/adam-optimization-algorithm/?fbclid=IwAR2PTzeRAXv3RxM6Oq0bRJ0cArOg1ccGdQ2-ASqRr4mXyj_WA-wUF3qG5Ag)
- Why Momentum Really Works - [link](https://distill.pub/2017/momentum/?fbclid=IwAR2KcWR77PqnGgW_9hv1vzcitx5XcqII_Jw1nuofqzN4IvxlZRGOMH9VJss)

### Chapter 5: Network Turning Strategy
#### Lecture
- Network Turning Strategy Overview
- Keras with TF 2.0 - [강의영상](https://vimeo.com/329503817/2c99e32970)

#### Coding materials
- Dataset API - [강의영상](https://vimeo.com/265236379/6184e59d88)
- TFrecordDataset - [강의영상#1](https://vimeo.com/267858109/c075bd8014), [#2](https://vimeo.com/267864427/6925a4b65b) [#3](https://vimeo.com/267913121/660dd2c986)

#### Reading materials
- [모델 튜닝법 - 남세동, 2018]https://www.facebook.com/dgtgrade/posts/1864002613658596
- [Tutorial on Keras flow_from_dataframe](https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c)
- [Keras on TPUs in Colab](https://medium.com/tensorflow/tf-keras-on-tpus-on-colab-674367932aa0?fbclid=IwAR0asD_jtPJIQr0ZGsAStziNcm3kQz4D0qAex5RjdG2uroQ24YuLGD0krqs)

#### Datasets
- Kuzushiji-MNIST - [arxiv](https://arxiv.org/abs/1812.01718?fbclid=IwAR04SwjXfrlVQxNiluqbrcS2-RKN8GTkMZyeIV5OMN-7liaVXPJZe23EUmY), [github](https://github.com/rois-codh/kmnist), [dataset](http://codh.rois.ac.jp/kmnist/?fbclid=IwAR2bv1XLI5YObvL0qrwIIi9eaLxPCSkV1pc8nJe3MyCaUUrjWqqjHXQZwtw)


### Chapter 6: Convolutional Neural Network
#### Lecture
- Convolution Nerual Net
- AlexNet
- Advanced CNN
    - VGG
    - Inception - GoogLeNet
    - ResNet

#### Additional materials
- Partial Convolution based Padding - [YouTube](https://www.youtube.com/watch?v=IKHzc7sGCxQ&feature=youtu.be&fbclid=IwAR3guScQRGjFOWuthjJbrrWP21K0mOt-fZTkFnIy95eQLhfbTHwZ0lxxPy8), [arxiv](https://arxiv.org/abs/1811.11718)


https://distill.pub/2019/memorization-in-rnns/?fbclid=IwAR3vo3wfvBiKTRw2b-aws0rsm9Uq7azG5lzpybJfI33e-La6y26GIecGILQ
<!-- ### Week x. Convolutional Neural Net
- Fundamentals of CNN
- Convolution & Pooling
- AlexNet
https://www.facebook.com/dgtgrade/posts/1592507747474752

### Week x. Performance Issues and Turning
- Data augumentation
https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3?fbclid=IwAR2pz0Zwc9PoC6D9FkYScqi7SuAVyJlNlheNecaRreaZGVrGBA7dYfTBTN4
- https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/ko/?fbclid=IwAR3l0i9-Fg5jNvfDB9RxcrbuEHJKtpp5pKrs4JO6j6MgG03VyoDw84rG10s#2 -->
-


<!-- ### Week x. Advanced CNN
### Week x. Neural Embeddings
- Concepts of Distributed Word Representation
- Word2Vec
- Skip-gram & CBOW
- Negative Sampling
https://monkeylearn.com/blog/beginners-guide-text-vectorization/?fbclid=IwAR0KOmSb-aOnuZMuVkciqwaH8CuzyoKggs4OCnHUZ8q6NpnOq66BtxauKdU
https://www.facebook.com/GoodDayToPlay/posts/375335199569148
https://brunch.co.kr/@goodvc78/16?fbclid=IwAR1bbPk29ngjr4esd3TuncLUiawhau9vcGMsVDpuUteZVE-cQeNjYvE2nv8

#### Reading materials
##### Papers
##### Articles
- [CBOW Impementation wit Keras](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)

##### Videos


### Week x. Recurrent Neural Net
- Fundamentals of RNN
- LSTM (Long Short-Term Memory model)
- GRU (Gated Recurrent Unit)
- Sequence2Sequence Model (Encoder-Decoder)
https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6?fbclid=IwAR3Lv1yFyhULobDVzSDHG_ctuvU3G9-vVSmpGA4Icp-SWeyTbXipm6IOifM
http://roboticist.tistory.com/571?fbclid=IwAR2McFuG4KUpMhDAmtRLx1kV90PvrR9Uozqnwrj8k1LUWfrTmypQ9cz2IW0

https://www.tensorflow.org/tutorials/sequences/text_generation?fbclid=IwAR0DqihLFK5NbpFacg_qHQPL5lwYQn8sxlnoS3yz1sopJzjzxkdqeG_EQiE


### Week x. Learning Representation for unsupervised learning
- Deep Belif Network
- Autoencoder
- DEC
- ~~
https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511?fbclid=IwAR0KKuBf-RNQwbbTBH1J47kPyRbO_FLKo3F6gQMEcW0b_pPFCsv4_CLoLyQ
https://www.youtube.com/watch?v=6DO_jVbDP3I&fbclid=IwAR3gdGJhYi_jmLInWTBhTPDNyi1ysh1PS-TkUkAT7yx6jURpAu54zqX34NE


#### Week x. Advanced Applications
- Neural Style
- Object Detection
  - RCNN, Fast(er)-RCNN
  - U-net
  - YOLO


Object Detection for Dummies - [#1](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html), [#2](https://lilianweng.github.io/lil-log/2017/12/15/object-recognition-for-dummies-part-2.html), [#3](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html), [#4](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)

#### Week x. Advanced Applications
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/?fbclid=IwAR2xCuq9fANQDQqGsz-0mKQ9zeOVUcOMs6cV6K-Yvmxg_p_34BiscaiSlGM
  - Semantic Segmentation
  - FCN, DeconvNet, DeepLab
  - U-Net, Fusion Net, PSPNet

https://weiminwang.blog/2017/03/20/using-tensorflow-to-build-image-to-text-deep-learning-application/?fbclid=IwAR3WcTz_LViqYdxh01yiQid3wN44uH8Vu4MUA1zCobrRuvThTt4eXpuHK4o
https://github.com/hccho2/Tacotron-Wavenet-Vocoder?fbclid=IwAR30GuMX73FtupaNmwRrSftf9hAELMmqFJyecnGz1xYl_ieeMDhSTmEsB7s


#### Week x. Genearive Models
- Variational Autoencoder
- Generative Adversarial Network
- Variants of GAN
https://www.slideshare.net/MingukKang/generative-adversarial-network-89571268?fbclid=IwAR0QR37PJ3hOk_xIiFG7zSi0Pk8_e_mpkySRWAJfdQx0kPgp5pAS-aGM9ys


From GAN to WGAN - [link](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

#### Week x. Modern Issues
https://www.facebook.com/groups/TensorFlowKR/permalink/554019594939103/ -->

## References

#### Reading Materials
- 딥 러닝 개발 면접 질문 (남세동, 2018) - [link](https://www.facebook.com/dgtgrade/posts/1679749038750622)
- 딥 러닝 공부법 모음 (남세동, 2018) - [link](https://www.facebook.com/dgtgrade/posts/1340177956041067)
- 딥 러닝 프로젝트 목록 (박규병, 2018) - [link](https://www.facebook.com/groups/TensorFlowKR/permalink/759277004413360/)

#### Lecture Materials
- 딥러닝 책 정리 자료 (이활석, 2018), [link](https://www.facebook.com/groups/TensorFlowKR/permalink/451098461897884/)
- Tesnsorflow Online Handbook - [link](https://www.facebook.com/MontrealAI/posts/723743667970894)
- Deep Learning Papers Reading Roadmap - [link](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap?fbclid=IwAR3TTzaMIvHKshrGntLAx5cL8Mdh-cQngp5qhhrYJGVGF0EUhtUnL134dxQ)
-  딥 러닝 공부 자료 모음 (고재형, 2017) - [link](https://www.facebook.com/groups/TensorFlowKR/permalink/556980944642968/)
- 딥 러닝 관련 페이스북 아티클 모음 (이활설, 2017) - [link](https://www.facebook.com/groups/TensorFlowKR/permalink/490430184631378/)
- TensorFlow and DeepLearning without Ph.D - [link](https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?fbclid=IwAR0qsGtfunN-FpUj5Fhhy6TezYVSZMvsJThAOGuMkoJ35FQGT7PSVzQcOkc&slide=id.g110257a6da_0_476)


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
https://www.analyticsvidhya.com/blog/2018/12/best-data-science-machine-learning-projects-github/?fbclid=IwAR2VtTqnYCo4AsFBOAzwcp2f4dbB4Mu_R0FtJwkYV3aKzdqN5Z6pLv0Zl0E
https://arxiv.org/pdf/1412.3555.pdf
https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/43905.pdf
https://arxiv.org/pdf/1712.00170.pdf
https://arxiv.org/abs/1406.2661 -->
