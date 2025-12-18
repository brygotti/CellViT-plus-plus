# Adapting pre-trained encoder backbones for CellViT
This repository is a fork of [CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus) that adds two backbone architectures to the framework: Hibou-L and mmVirtues.

## Docker image
A Docker image with all dependencies installed is available in the `rcp_image` directory. It is tailored for EPFL's [RCP cluser](https://wiki.rcp.epfl.ch/en/home/CaaS). To build the image, be sure to provide your own user name, user ID, group name and group ID via the `LDAP_USERNAME`, `LDAP_UID`, `LDAP_GROUPNAME` and `LDAP_GID` build arguments, respectively. For example:
```bash
cd rcp_image
docker buildx build --tag=registry.rcp.epfl.ch/<your_image_name>:latest \
    --platform=linux/amd64 \
    --build-arg LDAP_USERNAME=<your_username> \
    --build-arg LDAP_UID=<your_uid> \
    --build-arg LDAP_GROUPNAME=<your_groupname> \
    --build-arg LDAP_GID=<your_gid> \
    --push .
```

> [!WARNING]
> The full build process may take up to 3 hours to complete due to the installation of various dependencies. The resulting image will be approximately 30GB in size, so ensure you have sufficient storage space available, as well as network bandwidth for pushing the image to the registry.

## Runing the image on RCP
Once the image is built and pushed to the RCP registry, you can run it on the cluster using the following command:
```bash
runai submit \
  --name cellvit \
  --image registry.rcp.epfl.ch/<your_image_name>:latest \
  --pvc <your_scratch_volume>:/scratch \
  --large-shm \
  --backoff-limit 0 \
  --run-as-gid <your_gid> \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity
```

For more information on how to use Run:AI on RCP, please refer to the [official documentation](https://wiki.rcp.epfl.ch/en/home/CaaS).

## Running the 3-fold cross-validation with the different backbones
In `logs/PanNuke/CellViTHV`, you can find the configuration files used to run each of the 3 folds with the SAM-H, Hibou-L and mmVirtues backbones.
* SAM-H: `logs/PanNuke/CellViTHV/SAM-H-reproduced/Fold-{1,2,3}/config.yaml`
* Hibou-L: `logs/PanNuke/CellViTHV/Hibou-L/Fold-{1,2,3}/config.yaml`
* mmVirtues: `logs/PanNuke/CellViTHV/mmVirtues/Fold-{1,2,3}/config.yaml`

To run a specific fold with a specific backbone, use the following command:
```bash
conda activate tissuevit
python cellvit/train_cellvit.py --config logs/PanNuke/CellViTHV/<backbone>/<fold>/config.yaml
```

Logs of the run as well as the trained weights will be saved in the corresponding fold directory.

## Dataset and pre-trained weights
The PanNuke dataset, processed directly to run with the CellViT++ framework, as well as the pre-trained weights for Hibou-L can be downloaded from [Google Drive](https://drive.google.com/drive/folders/18HtCXGqEvQw1zSHFpPGicZ8PyVGjFKce?usp=sharing). The mmVirtues pre-trained weights cannot be shared currently.

Depending on where you extract them on your system, you might need to update the paths in the configuration files accordingly, before you can run the training. For example:
```yaml
data:
  dataset_path: <path_to_extracted_PanNuke_dataset>
model:
  pretrained_encoder: <path_to_extracted_pretrained_weights>
```

