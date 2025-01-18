# GAN MNIST Repository

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) trained on the MNIST dataset. The GAN consists of two neural networks: a generator and a discriminator. The generator creates new images that resemble the MNIST digits, while the discriminator tries to distinguish between real MNIST images and those created by the generator.

## Requirements

To run the code, you will need:

- Python 3.x
- PyTorch
- torchvision
- numpy
- argparse
- os

You can install the required packages using pip:

```bash
pip install torch torchvision numpy
```

## Running the Code

To train the GAN, you can run the `run.py` script with the desired parameters. Here are the available arguments:

```bash
--n_epochs: number of epochs of training (default: 50)
--batch_size: size of the batches (default: 64)
--lr: learning rate for Adam optimizer (default: 0.0002)
--b1: decay of first order momentum of gradient for Adam optimizer (default: 0.5)
--b2: decay of second order momentum of gradient for Adam optimizer (default: 0.999)
--n_cpu: number of cpu threads to use during batch generation (default: 2)
--latent_dim: dimensionality of the latent space (default: 100)
--img_size: size of each image dimension (default: 28)
--channels: number of image channels (default: 1)
--sample_interval: interval between image samples (default: 500)
```

For example, to train the model for 100 epochs with a batch size of 128, you can use:

```bash
python run.py --n_epochs 100 --batch_size 128
```

The training process will output the loss values for the discriminator and generator, as well as the average scores for real and fake images.

## Directory Structure

- `./images/gan/`: This directory will contain the generated images saved during training.
- `./save/gan/`: This directory will contain the saved models (generator and discriminator) after training.
- `./datasets/mnist/`: This directory will contain the downloaded MNIST dataset.

## Notes

- Ensure that you have enough GPU memory to train the model, especially if you use a large batch size.
- The `save_image` function from `torchvision.utils` is used to save the generated images. These images will be normalized to be in the range [0, 1].
- The generator and discriminator models are saved using `torch.save`. You can load these models later for inference or further training.

## Acknowledgments

This implementation is based on one online tutorial(''https://blog.csdn.net/qq_39547794/article/details/125389000''). 
