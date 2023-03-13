<div align="center">

# CPMNetv2: A Simpler and Stronger 3D Object Detection Benchmark in Medical Image

![Python](https://img.shields.io/badge/python-3.7+-orange)
![Pytorch](https://img.shields.io/badge/torch-1.13.1+-blue)
![CUDA](https://img.shields.io/badge/cuda-11.6+-green)  
</div>  



## Benchmarks
### Luna
<div align="center">

| Methods                      | 1/8   | 1/4   | 1/2   | 1     | 2     | 4     | 8     | CPM   |  TTA  |
|:----------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
Dou et al.   (2017a)           | 0.692 | 0.745 | 0.819 | 0.865 | 0.906 | 0.933 | 0.946 | 0.839 | False |
Zhu et al.   (2018)            | 0.692 | 0.769 | 0.824 | 0.865 | 0.893 | 0.917 | 0.933 | 0.842 | False |
Wang et al.  (2018)            | 0.676 | 0.776 | 0.879 | 0.949 | 0.958 | 0.958 | 0.958 | 0.878 | False |
Ding et al.  (2017)            | 0.748 | 0.853 | 0.887 | 0.922 | 0.938 | 0.944 | 0.946 | 0.891 | False |
Khosravan et al. (2018)        | 0.709 | 0.836 | 0.921 | 0.953 | 0.953 | 0.953 | 0.953 | 0.897 | False |
Liu et al. (2019)              | 0.848 | 0.876 | 0.905 | 0.933 | 0.943 | 0.957 | 0.970 | 0.919 | False |
Song et al. (2020)             | 0.723 | 0.838 | 0.887 | 0.911 | 0.928 | 0.934 | 0.948 | 0.881 | False |
nnDetection v0.1               | 0.812 | 0.885 | 0.927 | 0.950 | 0.969 | 0.979 | 0.985 | 0.930 |**True** |
CPMNet v2 (ours)               | 0.896 | 0.939 | 0.961 | 0.962 | 0.972 | 0.981 | 0.981 | **0.956** | False |
*Methods with FPR*<sup>*</sup> |  --   |  --   |  --   |  --   |  --   |  --   |  --   |   --   | -- |
Cao et al. (2020) + FPR        | 0.848 | 0.899 | 0.925 | 0.936 | 0.949 | 0.957 | 0.960 | 0.925 | False |
Liu et al. (2019) + FPR        | 0.904 | 0.914 | 0.933 | 0.957 | 0.971 | 0.971 | 0.971 | 0.952 | False |
</div> 
<sup>*</sup> False Positive Reduction Network (second stage).



### Private data  
Both aneurysm detection in TOF-MRA and rib fracture detection in CT Scans, our method achieved better results (w.o TTA) than nndetection (w. TTA). You are welcome to use it in other public data sets and can update its performance to me.


# Installation
## Create conda env  
```bash
conda create -n env_name python==3.7
```
## Install requirements
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install SimpleITK==2.2.1 pandas==1.3.5 scikit-image==0.19.3 scipy==1.7.3
```

# Train  
```bash
bash train_xxx.sh
```
***Note:*** args.num-sam depend on the average number of instance (lesion) in per sample (N), suggest you set to 2N. The real batch size is args.batch_size * args.num-sam, be careful with your GPU memory.

**If you use CPMNetv2, please cite our papers:**
    
    {@inproceedings{song2020cpm,
    title={CPM-Net: A 3D Center-Points Matching Network for Pulmonary Nodule Detection in CT Scans},
    author={Song, Tao and Chen, Jieneng and Luo, Xiangde and Huang, Yechong and Liu, Xinglong and Huang, Ning and Chen, Yinan and Ye, Zhaoxiang and Sheng, Huaqiang and Zhang, Shaoting and others},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={550--559},
    year={2020},
    organization={Springer}
    }
    
    @article{luo2021scpmnet,
    title={SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and center points matching},
    author={Luo, Xiangde and Song, Tao and Wang, Guotai and Chen, Jieneng and Chen, Yinan and Li, Kang and Metaxas, Dimitris N and Zhang, Shaoting},
    journal={Medical Image Analysis},
    volume={75},
    pages={102287},
    year={2022},
    publisher={Elsevier}
    }



