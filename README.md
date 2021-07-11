# ID-Unet: Iterative-view-synthesis(CVPR2021 Oral)
Tensorflow implementation of [**ID-Unet: Iterative Soft and Hard Deformation for View Synthesis**](https://arxiv.org/pdf/2103.02264).

**The network architecture**
<p align="center"> <img src="./results/overview.png" width="55%"><br><center></center></p>
<p align="center"> <img src="./results/architecture.png" width="95%"><br><center></center></p>
<p align="center"> <img src="./results/CDM.png" width="95%"><br><center></center></p>

## Experiment Results
- chair
<p align="center"> <img src="./results/result1.png" width="55%"><br><center></center></p>
<p align="center"> <img src="./results/result3.png" width="95%"><br><center></center></p>

- MultiPIE
<p align="center"> <img src="./results/result2.png" width="95%"><br><center></center></p>
<p align="center"> <img src="./results/result4.png" width="95%"><br><center></center></p>

- Flow
<p align="center"> <img src="./results/flow.png" width="95%"><br><center></center></p>


## Preparation

- **Prerequisites**
    - Tensorflow 
    - Python 2.x with matplotlib, numpy and scipy
- **Dataset**
    - [3D chair](https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar) 
    - [MultiPIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)
	- Images should be placed in ./datasets/
- **Download model**
    - [Tool model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
        - model should be placed in ./models/
## Quick Start

Exemplar commands are listed here for a quick start.
### dataset
- prepare dataset

    ```console
    python dataset.py --path 'Path to Training Dataset'

### Training
- To train with size of 128 X 128

    ```console
    python  
    ```

### Testing
- Example of generating samples

    ```console
    python  
    ```

## Citation
If this work is useful for your research, please consider citing:

@inproceedings{yin2021id,
  title={ID-Unet: Iterative Soft and Hard Deformation for View Synthesis},
  author={Yin, Mingyu and Sun, Li and Li, Qingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7220--7229},
  year={2021}
}





