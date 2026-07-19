# CS231n Assignment 3 (Spring 2022)

This directory contains the official Spring 2022 Assignment 3 starter plus completed implementations and written answers. The archived course page is available at <https://cs231n.github.io/assignments2022/assignment3/>.

The original starter archive was `assignment3_colab.zip`, downloaded from the official course site. Its SHA-256 digest is:

```text
1606b0a8460407511769d2ab920ffbd6d84b1031a7bc76b3f2a016af7d77b4e4
```

## Contents

- `RNN_Captioning.ipynb`: vanilla RNN layers and COCO image captioning.
- `Transformer_Captioning.ipynb`: positional encoding, masked multi-head attention, and Transformer captioning.
- `Generative_Adversarial_Networks.ipynb`: vanilla GAN, LSGAN, and DCGAN on MNIST.
- `Self_Supervised_Learning.ipynb`: SimCLR augmentations, contrastive loss, and linear evaluation.
- `LSTM_Captioning.ipynb`: optional LSTM captioning exercise, completed here.

The core implementations live under `cs231n/` and are designed to be exercised by the checks embedded in the notebooks.

## Data

The course datasets and pretrained weights are intentionally not committed. From this directory, COCO captioning data can be downloaded with:

```bash
cd cs231n/datasets
bash get_datasets.sh
```

The GAN notebook downloads MNIST through `torchvision`. The self-supervised notebook downloads CIFAR-10 and its provided SimCLR checkpoint when run in Colab.

## Validation

The deterministic checks were run locally against the supplied reference files:

- RNN and LSTM step and sequence gradients.
- Fixed RNN and LSTM captioning losses.
- GAN/LSGAN losses, model parameter counts, and output shapes.
- SimCLR naive and vectorized losses and similarity matrices.
- Transformer shape, masking, causality, and gradient checks.

The 2022 notebooks pin behavior from an older PyTorch release. Exact seeded Transformer tensors can differ on current PyTorch versions because module initialization and dropout random-number streams changed; the implementation follows the specified equations and is additionally checked using version-independent invariants.

Run the dataset-free validation from this directory with:

```bash
python validate.py
```
