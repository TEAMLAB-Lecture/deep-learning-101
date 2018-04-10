### 파이썬 가상환경 설정
```bash
conda create -n dl python=3.6
activate dl
```

### 파이썬 머신러닝 에코시스템 설치
```bash
conda install -y jupyter          #Machine Learning toolkits
conda install -y scipy
conda install -y pandas
conda install -y numpy
conda install -y scikit-learn    
# conda install -c anaconda graphviz
conda install -y seaborn
conda install -y matplotlib                
conda install -y graphviz           #Visualization
```
#### Install GRAPHVIZ_DOT
Download & install
- https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi

### 파이썬 머신러닝 에코시스템 설치
```bash
conda install -y tensorflow
conda install -c conda-forge keras
```

### Keras Setup
```bash
where python # See Your
cd %USERPROFILE%\AppData\Local\conda\conda\envs\dl\etc\conda\activate.d
atom keras_activate.bat
```


#### Add `KERAS_BACKEND` setup in `keras_activate.bat`
```bash
set "KERAS_BACKEND=tensorflow"
set "GRAPHVIZ_DOT=C:\Program Files (x86)\Graphviz2.38\bin\dot.exe"
set "PATH=%PATH%;C:\Program Files (x86)\Graphviz2.38\bin"
```
