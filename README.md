# Approximate Rank-Order Clustering (AROC)
Implementation of Approximate Rank-Order Clustering (AROC) algorithm in [Clustering Millions of Faces by Identity](https://arxiv.org/abs/1604.00989). Features used in the implementation can be found at [face_verification_experiment by AlfredXiangWu](https://github.com/AlfredXiangWu/face_verification_experiment/raw/master/results/LightenedCNN_C_lfw.mat). Features extracted by other models can be used too.

## How to run
```bash
mkdir data

# Download the LFW features from [face_verification_experiment by AlfredXiangWu]
wget https://github.com/AlfredXiangWu/face_verification_experiment/raw/master/results/LightenedCNN_C_lfw.mat -P data

# Perform the clustering
python main.py
```
