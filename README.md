# Neural Graph  Collaborative Filtering with MovieLens



## Dependency

```java
pytorch >= 1.12.0
python >= 3.8
scipy >= 1.7.1
numpy >= 1.20.3
```

## Running the code

### 1.local

```java
python3 main.py -e 10 -b 256 -dl true -k 10 -fi 1m
```

### 2.online(recommend)

```python
!pip3 install mxnet-mkl==1.6.0 numpy==1.23.1
```

```
!git clone -b master https://github.com/CristiZZzz27/NGCF.git
```

```
cd NGCF
```

```python
!python3 main.py -e 10 -b 256 -dl true -k 10 -fi 1m
```

