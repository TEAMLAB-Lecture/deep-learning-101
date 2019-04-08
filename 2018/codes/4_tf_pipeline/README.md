## Lab assginemnt - TFRecords 만들기

이번 Lab은 TFRecords 를 생성해서 학습을 합니다. 주요 스펙은 아래와 같습니다.

1. Dog Breed 데이터 셋을 사용해서 TFRecords 데이터 셋을 만들 것
2. 만들어진 Dogdataset은 RGB 컬러를 Grayscale로 변환하여 저장해야함
  - 참고 URL - https://www.tensorflow.org/api_guides/python/image#Converting_Between_Colorspaces
3. Dogdataset에서 사진에 위치해 있는 강아지들만 따로 뽑아서 32by32의 크기로 resize할 것
  - 사진내 강아지의 위치가 담긴 Annotation 정보는 XML 형태로 제공됨 (Appendix 참고)
4. 32 by 32의 사진들을 flattern 하여 MLP을 분류하는 학습 코드를 작성할 것
5. 제출은 github url을 통해 싸이버 캠퍼스에 업로드 할 것

### Dataset
- Stanford Dogs Dataset(http://vision.stanford.edu/aditya86/ImageNetDogs/)
  - Images - http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
  - Annotation - http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

### Appendix
#### Annotation XML
```XML
<annotation>
	<folder>02086646</folder>
	<filename>n02086646_45</filename>
	<source>
		<database>ImageNet database</database>
	</source>
	<size>
		<width>500</width>
		<height>380</height>
		<depth>3</depth>
	</size>
	<segment>0</segment>
	<object>
		<name>Blenheim_spaniel</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>24</xmin>
			<ymin>20</ymin>
			<xmax>465</xmax>
			<ymax>355</ymax>
		</bndbox>
	</object>
</annotation>
```
