# DRPL: Deep Regression Pair Learning for Multi-Focus Image Fusion
This is the implementation of [DRPL:Deep Regression Pair Learning for Multi-Focus Image Fusion](https://ieeexplore.ieee.org/abstract/document/9020016). 

by Jinxing Li*; Xiaobao Guo*; Guangming Lu; Bob Zhang; Yong Xu; Feng Wu; David Zhang.

In this repo, we provide source codes and our dataset for the easily training and test.

###Abstract:
In this paper, a novel deep network is proposed for multi-focus image fusion, 
named Deep Regression Pair Learning (DRPL). In contrast to existing deep fusion
 methods which divide the input image into small patches and apply a classifier 
 to judge whether the patch is in focus or not, DRPL directly converts the whole 
 image into a binary mask without any patch operation, subsequently tackling the difficulty 
 of the blur level estimation around the focused/defocused boundary. Simultaneously, 
 a pair learning strategy, which takes a pair of complementary source images as inputs and 
 generates two corresponding binary masks, is introduced into the model, greatly imposing 
 the complementary constraint on each pair and making a large contribution to the performance 
 improvement. Furthermore, as the edge or gradient does exist in the focus part while there is 
 no similar property for the defocus part, we also embed a gradient loss to ensure the generated 
 image to be all-in-focus. Then the structural similarity index (SSIM) is utilized to make a
 trade-off between the reference and fused images. Experimental results conducted on the 
 synthetic and real-world datasets substantiate the effectiveness and superiority of DRPL 
 compared with other state-of-the-art approaches.
 
## Installation
- Install [PyTorch-0.4.0](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support PyTorch-0.4.0 and Python 3+. You may make some modification in the code to update it to higher version.
- Other dependencies: opencv-python
- Our pytorch_ssim is developed based on: (https://github.com/Po-Hsun-Su/pytorch-ssim)

## Dataset

We provide raw datasets and templates that synthesis our training data, as well as the preprocessing code.

For processed data, please download from this link:
https://drive.google.com/drive/folders/1C-djx2JUoVKWx4H_w55IgHdQKnPvuuAL?usp=sharing

or
https://pan.baidu.com/s/1OJDX4JlvL3OsrCrHl50nLg passwd: a78b

- We generate the synthetic images based on the raw images from the ImageNet Large Scale Visual 
Recognition Challenge 2012 (ILSVRC2012). Refer to /data/selected, all-in-focus images are cropped
 from raw data. More details can be found in our paper.

- For evaluation, refer to /data/sampleval100 as an example subset.
- Follow by our paper and code, you can generate your own dataset.

## Training and validation:

After setting up data and training environment, you can simply run:

```bash
cd DRPL
# by default, it runs on the GPU
# for best results, use default hyperparams in train_net.py
python train_net.py --train_path ./data/train_raw_blur_pair_20k.pkl
 --valid_path ./data/val_pair.pkl --test_path ./data/lytro
```
- Note that for lytro or other datasets, please download from related website.

## Testing:
```bash
cd DRPL
python test_net.py --load_ckpt ./model/your_trained_model.pkl
```
or run 
```
sh test.sh 
```
to test our provided model.

### Note:
Some parts in the source code are used for evaluation or prepocessing, 
which can be ignored in training or testing. More details please refer
to the code.

## Citation
If you use this code for your research, please cite our paper. For commercial use, please contact us.
```
@article{li2020drpl,
  title={DRPL: Deep Regression Pair Learning for Multi-Focus Image Fusion},
  author={Li, Jinxing and Guo, Xiaobao and Lu, Guangming and Zhang, Bob and Xu, Yong and Wu, Feng and Zhang, David},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4816--4831},
  year={2020},
  publisher={IEEE}
}
```