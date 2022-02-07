from src.gan import Generator, Discriminator
from src.utils import get_noise
import torch


def get_disc_loss(gen: Generator, disc: Discriminator, criterion: torch.nn.functional, real: torch.Tensor,
                  num_images: int, z_dim: int, device: str) -> torch.Tensor:
    """
    Return the loss of the discriminator given inputs.

    :param gen:         the generator model, which returns an image given z-dimensional noise .
    :param disc:        the discriminator model, which returns a single-dimensional prediction of real/fake.
    :param criterion:   the loss function, which should be used to compare the discriminator's predictions to the
                        ground truth reality of the images (e.g. fake = 0, real = 1)
    :param real:        a batch of real images
    :param num_images:  the number of images the generator should produce, which is also the length of the real images
    :param z_dim:       the dimension of the noise vector, a scalar
    :param device:      the device type
    :return:            a torch scalar loss value for the current batch
    """

    noise = get_noise(num_images, z_dim, device=device)
    fake_image = gen(noise)
    disc_fake_pred = disc(fake_image.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)
    return disc_loss


def get_gen_loss(gen: Generator, disc: Discriminator, criterion: torch.nn.functional, num_images: int, z_dim: int,
                 device: str) -> torch.Tensor:
    """
    Return the loss of the generator given inputs.

    :param gen:         the generator model, which returns an image given z-dimensional noise .
    :param disc:        the discriminator model, which returns a single-dimensional prediction of real/fake.
    :param criterion:   the loss function, which should be used to compare the discriminator's predictions to the
                        ground truth reality of the images (e.g. fake = 0, real = 1)
    :param real:        a batch of real images
    :param num_images:  the number of images the generator should produce, which is also the length of the real images
    :param z_dim:       the dimension of the noise vector, a scalar
    :param device:      the device type
    :return:            a torch scalar loss value for the current batch
    """

    noise = get_noise(num_images, z_dim, device)
    fake_image = gen(noise)
    disc_fake_pred = disc(fake_image)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss