import torch
from torch import nn

torch.manual_seed(0)


def get_generator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Function for returning a block of the generator's neural network
    given input and output dimensions.

    :param input_dim:       the dimension of the input vector, a scalar
    :param output_dim:      the dimension of the output vector, a scalar

    :return:                a generator neural network layer, with a linear transformation
                            followed by a batch normalization and then a relu activation
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 10, image_dim: int = 784, hidden_dim: int = 128):
        """
        Generator class constructor.

        :param noise_dim:   the dimension of the noise vector, a scalar
        :param image_dim:   the dimension of the images, fitted for the dataset used, a scalar
                            (MNIST images are 28 x 28 = 784 so that is your default)
        :param hidden_dim:  the inner dimension, a scalar
        """

        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(noise_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, image_dim),
            nn.Sigmoid()
        )

    def forward(self, noise: torch.Tensor) -> nn.Sequential:
        """
        Function for completing a forward pass of the generator.

        :param noise:       noise tensor with dimensions (n_samples, z_dim)
        :return:            generated images
        """
        return self.gen(noise)

    def get_gen(self) -> nn.Sequential:
        """
        :return:            the sequential model
        """
        return self.gen