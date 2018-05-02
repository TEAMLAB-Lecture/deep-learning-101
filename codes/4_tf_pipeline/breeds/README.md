## Handling images with TF data pipeline
본 repository는 Tensorflow를 사용하여, 이미지 데이터의 분류를 CNN으로 구현한 예시입니다. 본 예제는 아래와 같은 구현 예시를 보여줍니다.

- 이미 분류가 완료된 이미지 데이터를 사용하여 TFRecoder Datatype(protobuf) 생성함
- 생성된 TFRecorder 사진들을 TF.images 모듈을 활용하여 다양한 형태로 변형하여 학습함
  - 그림제외 0 padding
  - 사진 크기 맞춤
  - 색깔 변형
  - 좌우 대칭 변형 (랜덤)
- 실험 모델을 설정하여, CNN의 성능을 비교하는 시험을 실시함
- 실험 결과는 TensorBoard를 활용하여 데이터를 분석함

### 데이터
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  - Number of categories: 120
  - Number of images: 20,580
  - Annotations: Class labels, Bounding boxes
- Khosla, Aditya, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li. "Novel dataset for fine-grained image categorization: Stanford dogs." In Proc. CVPR Workshop on Fine-Grained Visual Categorization (FGVC), vol. 2, p. 1. 2011.


### 실험 모델의 구성
- 실험 데이터
  - 전처리 미실시 버전
  - 전처리 처리 버전
- 학습 모델
  - Fully Connected (HyperParameter 튜닝)
  - CNN 기본 모델 (HyperParameter 튜닝)
  - AlexNet (HyperParameter 튜닝)
  - ResNet (HyperParameter 튜닝)

### 데이터 분석 순서
1. 데이터 다운로드 한다
```bash
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar # 이미지 파일 다운로드
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar # Annotation 파일 다운로드
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar # Annotation 파일 다운로드 ```
2. 데이터의 압축을 푼다.
```bash
tar -xvf annotation.tar # Annotation 폴더가 생성됨
tar -xvf images.tar # Images 폴더가 생성됨

```

### 뉴 데이터 셋
https://hci.iwr.uni-heidelberg.de/node/6132
