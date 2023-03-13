<div align="center">

# CPMNetv2: A simple and efficient 3D object detection benchmark in medical image

![Python](https://img.shields.io/badge/python-3.7+-orange)
![Pytorch](https://img.shields.io/badge/torch-1.13.1+-blue)
![CUDA](https://img.shields.io/badge/cuda-11.6+-green)  

</div>  

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



