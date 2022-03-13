# AI Fall Detection - How to run

## Content

* [Requirement](#requirement)
  
  * [Software](#software)
  
  * [Library](#library)
    
    * [Installation](#installation)
  
  * [Dataset](#dataset)
  
  * [Source Code](#source-code)
  
  * [Clone from github](#clone-from-github)

* [Folder Structure](#folder-structure)

# 

## Requirement

### Software

* [Anaconda](https://www.anaconda.com/)

* [Spyder](https://www.spyder-ide.org/)



### Library

* [MediaPipe](https://google.github.io/mediapipe/)

* [TensorFlow](https://www.tensorflow.org/)

#### Installation

**Anaconda**

If you are windows user follow steps from this [link](https://docs.anaconda.com/anaconda/install/windows/).

**MediaPipe**

```shell
pip install mediapipe
```

**Tensorflow**

```shell
pip install tensorflow
```



### Dataset

Please download from this [link](https://drive.google.com/drive/folders/10rgr6mk7qBQfZjGZj610k1FPVQBdBMlQ?usp=sharing).



### Source Code

Please download `main.py` and `videorecorder.py` from this [link](https://drive.google.com/drive/folders/1Fjw_E1Si-6RMxMbPUfr6foi7spSACNcq?usp=sharing).



### Clone from github

via Terminal

```shell
git clone https://github.com/snoymy/AI_fall_detection.git
```

via web [link](https://github.com/snoymy/AI_fall_detection)

![](./assets/web.png)

# 

## Folder Structure

After install **libray** , **dataset** and **source code** , Please create project root directory `Ex. fall_detector` and place all file you had download in the folder

```
fall_detector
  ├── train_video
  │    ├── fall
  │    └── normal
  ├── main.py
  └── videorecorder.py
```

Open `main.py` with **Spyder** and press run.

![](./assets/run.png)



after run

![](./assets/main_program_run.png)



and predict

![](./assets/predict.png)

if you only want to train and test just run only `main.py`, but if you want collect new dataset run `videorecorder.py`.
