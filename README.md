# OCELOT2023
Solution of Team saltfish for OCELOT 2023: Cell Detection from Cell-Tissue Interaction

You can reproduce our method as follows step by step:


## Environments and Requirements

Our development environments:

| System                  | Ubuntu 22.04.1 LTS                         |
| ----------------------- | ------------------------------------------ |
| CPU                     | Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz  |
| RAM                     | 16*×*4GB; 2.67MT/s                         |
| GPU(number and type)    | 4 NVIDIA Titan RTX 24G                   |
| CUDA version            | 11.0                                       |
| Programming language    | Python 3.8.5                              |
| Deep learning framework | Pytorch (Torch 1.7.1, torchvision 0.8.2) |
| Specific dependencies   | monai 0.9.0                                |

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

-  [OCELOT 2023: Cell Detection from Cell-Tissue Interaction Challenge official dataset](https://ocelot2023.grand-challenge.org/datasets/).

## Preprocessing

- Preprocess for images and labels:

  ```
  python utils/data_preprocess.py -i <path_to_original_data> -o <path_to_processed_data>
  ```
## Training

To train the models, run this command :

  ```
  python train.py --data_path_tissue <path_to_processed_data_tissue> --data_path_cell <path_to_processed_data_cell>  --model_name "samh_unet_final"
  ```
Then we get two models saved in ./work_dir/samh_unet_final

## Trained Models
The publicly available pre-trained vit models can be download here:[Google Drive](https://drive.google.com/drive/folders/1UVwNHj9Y47j516SEUdtn1nlDau1kksDj)

You can download models trained on the above dataset with the above code here:

You can refer to [this](create_docker/user/unet_example) to generate a Docker file 

Docker  container link:[Docker Hub](https://hub.docker.com/repository/docker/zyleeustc/ocelot2023/general)

## Results on test set
  | Method | Complete model |
  | ------ | -------------- |
  | F1     |  0.7243        |
