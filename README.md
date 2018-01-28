TEAMLAB MOOC: Deep Learning - Theroy and Application
==================================

본 강의는 TEAMLAB의 데이터 과학 시리즈 MOOC의 일환으로 제작됩니다. 본 과정은 아래와 같이 구성됩니다.

- (K-MOOC) [데이터 과학을 위한 파이썬 입문](http://www.kmooc.kr/courses/course-v1:GachonUnivK+ACE.GachonUnivK01+2016_01/about)
- (Inflearn) [데이터 과학을 위한 파이썬 입문](https://www.inflearn.com/course/python-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9E%85%EB%AC%B8-%EA%B0%95%EC%A2%8C/) - 위와 동일과정이나 Lab Assignment등 업데이트 및 질의응답 제공
- (Inflearn)[밑바닥부터 시작하는 머신러닝](https://www.inflearn.com/course/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%9E%85%EB%AC%B8-%EA%B0%95%EC%A2%8C/)
- (K-MOOC)[Operation Research with Python Programming](https://github.com/TeamLab/Gachon_CS50_OR_KMOOC)
- [Applied Database System with Python Programming](https://github.com/TeamLab/database_101/blob/master/README.md) - 개발중
- [Deep Learning - Theroy and Application]() - 본 과정

## 강의 개요
딥 러닝의 시대!
80년대 Hilton 교수에 의해 제안된 Back propagation 구조로 인간의 신경망과 유사한 형태로 뉴럴넷을 구성하여 학습 가능하다는 것이 알려진 이후, 지난 20년간 뉴럴넷은 비약적인 발전을 거듭했다. 특히 2000년대 초반 정보화 혁명과 스마트폰의 출현이후 폭증가하는 데이터와 GPU의 발전 그리고 연구자들의 끊임없는 노력에 의해 개발된 새로운 알고리즘은 현재를 딥 러닝의 시대로 이끌어가고 있다. 본 강의는 딥러닝을 학습하기 위해 기본적인 알고리즘인 ANN, CNN, RNN을 학습하고, 이를 바탕으로 최근 가장 많은 관심을 받고 있는 VAE와 GAN의 기초이론을 학습한다. 또한 실제 Application에 적용하기 위해 대표적인 딥러닝 프레임워크인 Tesnsoflow와 Keras을 학습하여, 다양한 딥 러닝 Application을 개발하는 것을 목표로 한다.

## 강의 목표
- 딥 러닝의 구성하는 기초 이론과 알고리즘을 학습함
- 딥 러닝 구현을 위해 대표적인 딥러닝 프레임워크인 Tensorflow와 Keras를 학습함
- 딥 러닝을 활용한 다양한 Application과 관련 논문을 구현함

## 강의 정보

* 강좌명: Deep Learning - Theroy and Application
* 강의자명: 가천대학교 산업경영공학과 최성철 교수 (sc82.choi@gachon.ac.kr, Director of [TEAMLAB](http://theteamlab.io/))
* Facebook: [Gachon CS50](https://www.facebook.com/GachonCS50)
* Email: teamlab.gachon@gmail.com

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
  - PyTorch

#### Reading materials
- [History and Background](https://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html)

#### Supplements - Environment setup
- 가상환경과 Package 활용하기 - [강의영  상](https://www.youtube.com/watch?v=QLF5UvUvKCo&list=PLBHVuYlKEkUJvRVv9_je9j3BpHwGHSZHz&index=51), [강의자료](https://doc.co/SoCj3W/EFk5T6)
- Python ecosystem for machine learning - [강의영상](https://vimeo.com/247903638/96dc854a53), [강의자료](https://1drv.ms/b/s!ApZ4mg7k2qYhgaNZscJaF1fC63jl_Q)
- How to use Jupyter Notebook - [강의영상](https://vimeo.com/248135457/5047913a77), [강의자료](https://1drv.ms/b/s!ApZ4mg7k2qYhgaNaEydTqvLQIgXRCQ)


#### Supplements - Linear algebra
  - Lab: Simple Linear algebra concepts - [강의영상](https://vimeo.com/245942627/d2e4ef3e5e), [강의자료](https://1drv.ms/b/s!ApZ4mg7k2qYhgaMuKaE5x8t0z1Z4vw)
  - Lab: Simple Linear algebra codes - [강의영상](https://vimeo.com/245943473/7372cc35c3), [강의자료](https://1drv.ms/b/s!ApZ4mg7k2qYhgaMv7umjL_JYHsubsA)
  - Assignment: Linear algebra with pythonic code -  [PDF](https://github.com/TeamLab/introduction_to_python_TEAMLAB_MOOC/raw/master/lab_assignment/lab_bla/lab_bla.pdf), [강의자료](https://github.com/TeamLab/introduction_to_python_TEAMLAB_MOOC/tree/master/lab_assignment/lab_bla)

#### Supplements - Numpy
- Chapter Intro - [강의영상](https://vimeo.com/249674805/e3f21116ab), [강의자료](https://1drv.ms/b/s!ApZ4mg7k2qYhgaNyqXoFReKxEoauMA), [강의코드](https://github.com/TeamLab/machine_learning_from_scratch_with_python/tree/master/code/ch3), [코드다운로드](https://s3.ap-northeast-2.amazonaws.com/teamlab-gachon/mooc_pic/ml_ch3.zip)
- Numpy overview - [강의영상](https://vimeo.com/248743492/702e0b7cb9)
- ndarray - [강의영상](https://vimeo.com/248743595/2bb4044b0f)
- Handling shape - [강의영상](https://vimeo.com/248743660/f7556bf9f0)
- Indexing & Slicing - [강의영상](https://vimeo.com/249209302/5684d9c74d)
- Creation functions - [강의영상](https://vimeo.com/249209309/928603c39f)
- Opertaion functions - [강의영상](https://vimeo.com/249209319/6a91bf02e2)
- Array operations - [강의영상](https://vimeo.com/249209338/aa25a7d5fa)
- Comparisons - [강의영상](https://vimeo.com/249209348/2d08684423)
- Boolean & fancy Index - [강의영상](https://vimeo.com/249931252/08b426eceb)
- Numpy data i/o - [강의영상](https://vimeo.com/249931258/74a7d3812d)
- Assignment: Numpy in a nutshell -  [PDF](https://s3.ap-northeast-2.amazonaws.com/teamlab-gachon/mooc_pic/lab_numpy.pdf), [강의자료](https://github.com/TeamLab/machine_learning_from_scratch_with_python/tree/master/lab_asssigment/1_lab_numpy)

### Chapter 2: Neural Network Basic
#### Lecture
- Linear Regression
- Gradient descent
- Stochastic gradient descent
- Logistic Regression
- Softmax

### Chapter 3: Aritificial Neural Net - Algorithms & Implmentation
#### Lecture
- MLP (Multi-Layer Perceptron)
- Backpropagation

#### Reading materials
- [계산 그래프로 역전파 이해하기](https://brunch.co.kr/@chris-song/22)

### Chapter 4: Building blocks of Deep Learning Part I
#### Lecture
- Optimization Methods
- Normalization

#### Reading materials
- [자습해도 모르겠던 딥러닝, 머리속에 인스톨 시켜드립니다.](https://www.slideshare.net/yongho/ss-79607172), 하용호, 2017
- [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039), 2017

### Chapter 5: Building blocks of Deep Learning Part II
#### Lecture
- Dropout
- Regularization
- Weight Initialization

### Chapter 6: Convolutional Neural Network(CNN)
#### Lecture
- Fundamentals of CNN
- Convolution & Pooling
- Weakly Supervised Localization
- LeNet
- AlexNet

### Chapter 7: Image handling and Data augumentation
#### Lecture
- Image handling methods
- Numpy
- OpenCV
- TF
- Data augumentation

#### References
- [Image augmentation for machine learning experiments](https://github.com/aleju/imgaug)


### Chapter 8: Advanced CNN
#### Lecture
- VGGNet
- GoogLeNet
- ResNet
- Inception

### Midterm Project: Dogs vs Cats, Classification of Dog Bleeding problems

### Chapter 9: Embedding techniques & Dimention reduction Part I
#### Lecture
- Concepts of Distributed Word Representation
- Word2Vec
  - Skip-gram & CBOW
  - Negative Sampling

#### Reading materials
- [Neural Network Language Model](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/29/NNLM/), ratsgo's blog, 2017
- [Word2Vec의 학습 방식](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/), ratsgo's blog, 2017
- [빈도수 세기의 놀라운 마법 Word2Vec, Glove, Fasttext](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/), ratsgo's blog, 2017

### Chapter 10: Embedding techniques & Dimention reduction Part II
#### Lecture
- Doc2Vec
- GloVe
- Autoencoder

### Chapter 11: Recurrent Neural Network Basic (RNN)
#### Lecture
- Fundamentals of RNN
- Valila RNN
- LSTM (Long Short-Term Memory model)
- GRU (Gated Recurrent Unit)

### Chapter 12: Advanced Sequence Models
#### Lecture
- Sequence2Sequence Model (Encoder-Decoder)
- Attention Mechanism
- Transformer

### Chapter 13: Generative Models
#### Lecture
- Variational Autoencoder
- Generative Adversarial Network
- Variants of GAN

### Chapter 14: Modern papaers & applications - Part I
#### Lecture
- Neural Style
- Object Detection
  - RCNN, Fast(er)-RCNN
  - U-net
  - YOLO

### Chapter 15: Modern papaers & applications - Part II
#### Lecture
- Semantic Segmentation
  - FCN, DeconvNet, DeepLab
  - U-Net, Fusion Net, PSPNet
- Image Captioning
- Quetion and Answering
  - DCN
  - Bi-DAF



## Keywords
파이썬, python, 딥러닝, 딥러닝 입문, 딥러닝 강좌, Deep Learning, Deep Learning, 딥러닝 강의, Deep Learngng 강의, Deep Learning MOOC, 가천대 최성철, 최성철 교수, 데이터 과학, 데이터 사이언스, Data science
