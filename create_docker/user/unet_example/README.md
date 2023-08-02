# This is a guide on how to create a Docker file.

1. Replace `create_docker/Dockerfile` with `create_docker/user/unet_example/Dockerfile`. This is necessary to install PyTorch and other dependencies.
2. Replace `create_docker/requirements.txt` with `create_docker/user/unet_example/requirements.txt`.
3. Reference the new inference code from the program entry-point i.e. `create_docker/process.py`. Simply modify `L8` as follows:

```diff
- from user.inference import Model
+ from user.unet_example.unet import PytorchUnetCellModel as Model
``` 

4. Download our model weights from [here](https://drive.google.com/drive/folders/1oPUqhRfoY7c17CAuH8FHX0CguNrm_wI5), and put the file inside the directory `create_docker/user/unet_example/checkpoints/`.

5. Verify the code works by running a simple test:

```sh
bash test.sh
```

6.  Export the code:

```sh
bash export.sh
```
