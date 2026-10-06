# Fashion-MNIST: MLP and CNN

Completed AI473 academic model comparison, student F2023376179. Both architectures use the same seeded train/validation/test split. `results/metrics.json` and `results/comparison.csv` contain the fresh five-epoch, 20,000-training-pool run. The 10,000-example official test split is evaluated separately. The notebook reads these numeric results and model definitions without requiring missing files.

Install `requirements.txt`, then run `python src/main.py --epochs 5 --train-subset 20000`. Fashion-MNIST downloads locally from its official dataset repository with resource checksums. Generated model weights and plots remain local and are excluded from this media-free GitHub repository. See `VERIFICATION.json` for executed checks and limits.
