# EF-MVSNet

[HIGH-RESOLUTION MULTI-VIEW STEREO WITH DYNAMIC EDGE FLOW](https://ieeexplore.ieee.org/document/9428281). Kui Lin, Lei Li, Jianjun Zhang, Xing Zheng, Suping Wu.

## Environment
* python 3.6 (Anaconda)
* pytorch 1.0.1
* opencv-python 4.3.0

## Training

* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.
```

├── Cameras
├── Depths
    ├── scan
    ├── scan_train
├── Rectified

```
* in ``train.sh``, set ``MVS_TRAINING`` as your training data path
* Train EF-MVSNet:  ```./train.sh ```
## Testing

* Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the ``DTU_TESTING`` folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* in ``test.sh``, set ``DTU_TESTING`` as your testing data path and ``CKPT_FILE`` as your checkpoint file. You can also download my [pretrained model](https://drive.google.com/file/d/1UdV9Aey9yj0EgqwL2yZB1bDj_Hz9U2fM/view?usp=sharing).
* Test MVSNet: ``./test.sh``

## Fusion
DTU dataset: ```./eval_dtu.sh```

Tanks and Temples dataset: ```./eval_tank.sh```
## Results on DTU

|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| MVSNet(D=256)         | 0.396  | 0.527  | 0.462    |
| PyTorch-MVSNet(D=192) | 0.449  | 0.379  | 0.414    |
| EF-MVSNet(D=192)      | 0.402  | 0.375  | 0.388    |


## Acknowledgment
---
This repository is partly based on the [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch) repository by Xiaoyang Guo. Many thanks to Xiaoyang Guo for the great code!

This repository is inspired by the [MVSNet](https://github.com/YoYo000/MVSNet) by Yao Yao et al. Many thanks to Yao Yao and his mates for the great paper and great code!
