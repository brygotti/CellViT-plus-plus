Run training

From the repo root:

```bash
cd /scratch/users/misenta/CellViT-plus-plus
python3 cellvit/train_cellvit.py --config configs/examples/pannuke_mmvirtues.yaml --gpu 0
```

If you want to disable wandb logging, set in the config:

- `logging.mode: offline`