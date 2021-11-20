"""
Image transformations for model training and prediction
"""

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as tvtransforms


def random_rotate(im: np.ndarray) -> np.ndarray:
    width = im.shape[1]
    height = im.shape[0]

    sign = 1 if random.random() > 0.5 else -1
    theta = sign * 30 * random.random()

    M = cv2.getRotationMatrix2D((width // 2, height // 2), theta, 1)

    im = cv2.warpAffine(im.copy(), M, (width, height), flags=cv2.INTER_LINEAR)

    return im


def random_flip(im: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        im = np.fliplr(im)

    if random.random() > 0.5:
        im = np.flipud(im)

    return im.copy()  # produce new array rather than view


def normalize_image(im, new_min, new_max):
    im_max, im_min = im.max(), im.min()

    return (im - im_min) * (new_max - new_min) / (im_max - im_min) + new_min


def random_gamma(im):
    if random.random() < 0.4:
        return im

    gamma_low, gamma_high = 0.7, 1.3

    # operate across channels independently

    im_gammas = []

    for channel_idx in range(im.shape[2]):
        im_channel = im[..., channel_idx]

        im_max, im_min = im_channel.max(), im_channel.min()

        gamma = np.random.uniform(gamma_low, gamma_high)

        im_channel = normalize_image(
            normalize_image(im_channel, 0, 1) ** gamma, im_min, im_max
        )

        im_gammas.append(im_channel[..., np.newaxis])

    return np.concatenate(im_gammas, axis=-1)


def random_gauss_noise(im):
    sigma_max = 0.025

    sigmas = (
        np.array([0, 0, 0])
        if random.random() > 0.7
        else np.array([sigma_max, sigma_max, sigma_max])
        # else np.random.uniform(0, sigma_max, 3)
    )

    return np.clip(
        im
        + (
            sigmas[np.newaxis, np.newaxis]
            * np.random.randn(im.shape[0], im.shape[1], 3)
        ),
        0,
        None,
    )


def random_brightness(im):
    brightness = 0.4

    alphas = (
        np.array([1, 1, 1])
        if random.random() > 0.8
        else 1.0 + np.random.uniform(-brightness, brightness, 3)
    )

    return im * alphas[np.newaxis, np.newaxis]


aug_transform = tvtransforms.Compose(
    [
        lambda im: im.transpose((1, 2, 0)),
        random_rotate,
        random_flip,
        random_gamma,
        random_gauss_noise,
        random_brightness,
        lambda im: im.transpose((2, 0, 1)),
        lambda im: torch.as_tensor(im, dtype=torch.float32),
    ]
)

transform = tvtransforms.Compose(
    [lambda im: torch.as_tensor(im, dtype=torch.float32)]
)
