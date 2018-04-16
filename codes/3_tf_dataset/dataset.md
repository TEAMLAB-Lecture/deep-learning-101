- 데이터 셋을 만드는 두 가지 방법
  - Creating a `source` (e.g.Dataset.from_tensor_slices()) constructs a dataset from one or more tf.Tensor objects.
  - Applying a `transformation` (e.g. Dataset.batch()) constructs a dataset from one or more tf.data.Dataset objects.

- 데이터셋에서 추출
  - tf.data.Iterator는 데이터 세트에서 요소를 추출하는 주요 방법을 제공합니다.

- 입력 파이프라인을 시작하는 방법, 메모리의 일부 텐서 (tensors)에서 데이터 세트를 구성하려면
  - tf.data.Dataset.from_tensors () 또는
  - tf.data.Dataset.from_tensor_slices
  - 입력 데이터가 권장 TFRecord 형식의 디스크에있는 경우 tf.data.TFRecordDataset을 구성 할 수 있습니다.


  dataset1  ==> Iterator 를 만드는 구조를 가져가면됨
  
