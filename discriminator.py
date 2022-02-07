from torch import nn
import torch


def get_discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.

    :param input_dim:       the dimension of the input vector, a scalar
    :param output_dim:      the dimension of the output vector, a scalar
    :return:                a discriminator neural network layer, with a linear transformation
                            followed by an nn.LeakyReLU activation with negative slope of 0.2
                            (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    """

    return nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, image_dim=784, hidden_dim=128):
        """
        Discriminator Class constructor

        :param image_dim:   the dimension of the images, fitted for the dataset used, a scalar, default = 28x28
        :param hidden_dim:  the inner dimension, a scalar
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(image_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.

        :param image:       a flattened image tensor with dimension (image_dim)
        :return:
        """
        return self.disc(image)

    def get_disc(self) -> nn.Sequential:
        """
        :return:            the sequential model
        """
        return self.disc
