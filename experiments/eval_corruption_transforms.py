# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This source code is taken from https://github.com/facebookresearch/augmentation-corruption for c-bar transforms.
# The source code is taken from https://github.com/bethgelab/imagecorruptions for c corruptions, based on the original repo https://github.com/hendrycks/robustness
# This is to rework some deprecated functions in sk-image

import abc
import numpy as np
from scipy.fftpack import ifft2
from scipy.ndimage import gaussian_filter, rotate, zoom
from skimage.draw import line_aa
import imagecorruptions
import numpy as np
from PIL import Image

import math

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
import cv2
from scipy.ndimage.interpolation import map_coordinates
import importlib.resources as pkg_resources


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)




# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = zoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img

def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////

def gaussian_noise(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [0.04, 0.06, .08, .09, .10][severity - 1]
    elif scale == 'tin':
        c = [0.04, 0.08, .12, .15, .18][severity - 1]
    else:
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]


    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [500, 250, 100, 75, 50][severity - 1]
    elif scale == 'tin':
        c = [250, 100, 50, 30, 15][severity - 1]
    else:
        c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1, scale='in', rng=None):

    if rng is None: 
        #global worker rng with fixed seed to make this reproducible every time the Dataloader is initialized
        from data import fixed_worker_rng
        rng = fixed_worker_rng

    if scale == 'cifar':
        c = [.01, .02, .03, .05, .07][severity - 1]
    elif scale == 'tin':
        c = [.01, .02, .05, .08, .14][severity - 1]
    else:
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c, rng=rng)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [.06, .1, .12, .16, .2][severity - 1]
    elif scale == 'tin':
        c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]
    else:
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [.4, .6, 0.7, .8, 1][severity - 1]
    elif scale == 'tin':
        c = [.5, .75, 1, 1.25, 1.5][severity - 1]
    else:
        c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1, scale='in'):
    # sigma, max_delta, iterations
    if scale == 'cifar':
        c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]
    elif scale == 'tin':
        c = [(0.1,1,1), (0.5,1,1), (0.6,1,2), (0.7,2,1), (0.9,2,2)][severity - 1]
    else:
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
        severity - 1]

    x = np.uint8(
        gaussian(np.array(x) / 255., sigma=c[0]) * 255)
    x_shape = np.array(x).shape

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(x_shape[0] - c[1], c[1], -1):
            for w in range(x_shape[1] - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0,
                   1) * 255


def defocus_blur(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    elif scale == 'tin':
        c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]
    else:
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    if len(x.shape) < 3 or x.shape[2] < 3:
        channels = np.array(cv2.filter2D(x, -1, kernel))
    else:
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1, scale='in'):
    shape = np.array(x).shape
    if scale == 'cifar':
        c = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]
    elif scale == 'tin':
        c = [(10,1), (10,1.5), (10,2), (10,2.5), (12,3)][severity - 1]
    else:
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)

    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >=3:
            return np.stack([gray, gray, gray], axis=2)
        else:
            return gray
    else:
        return np.clip(x, 0, 255)


def zoom_blur(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    elif scale == 'tin':
        c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    else:
        c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)

    set_exception = False
    for zoom_factor in c:
        if len(x.shape) < 3 or x.shape[2] < 3:
            x_channels = np.array([x, x, x]).transpose((1, 2, 0))
            zoom_layer = clipped_zoom(x_channels, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
        else:
            zoom_layer = clipped_zoom(x, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

        try:
            out += zoom_layer
        except ValueError:
            set_exception = True
            out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

    if set_exception:
        print('ValueError for zoom blur, Exception handling')
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def fog(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]
    elif scale == 'tin':
        c = [(.4,3), (.7,3), (1,2.5), (1.5,2), (2,1.75)][severity - 1]
    else:
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
             :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    elif scale == 'tin':
        c = [(1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55)][severity - 1]
    else:
        c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]

    idx = np.random.randint(5)

    # Dynamically load the images from imagecorruptions library
    resource_name = f'frost{idx + 1}.png' if idx < 3 else f'frost{idx + 1}.jpg'

    with pkg_resources.open_binary('imagecorruptions.frost', resource_name) as file:
        file_data = file.read()

    image_array = np.frombuffer(file_data, np.uint8)
    frost = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    frost_shape = frost.shape
    x_shape = np.array(x).shape

    # resize the frost image so it fits to the image dimensions
    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[
        1]:  # If both dims are too small, pick the bigger scaling factor
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

    scaling_factor *= 1.1
    new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                 int(np.ceil(frost_shape[0] * scaling_factor)))
    frost_rescaled = cv2.resize(frost, dsize=new_shape,
                                interpolation=cv2.INTER_CUBIC)

    # randomly crop
    x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[
        0]), np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])

    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]]
        frost_rescaled = rgb2gray(frost_rescaled)
    else:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def snow(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]
    elif scale == 'tin':
        c = [(0.1,0.2,1,0.6,8,3,0.8),
         (0.1,0.2,1,0.5,10,4,0.8),
         (0.15,0.3,1.75,0.55,10,4,0.7),
         (0.25,0.3,2.25,0.6,12,6,0.65),
         (0.3,0.3,1.25,0.65,14,12,0.6)][severity - 1]
    else:
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0],
                                  scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)


    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    # The snow layer is rounded and cropped to the img dims
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, x.reshape(x.shape[0],
                                                            x.shape[
                                                                1]) * 1.5 + 0.5)
        snow_layer = snow_layer.squeeze(-1)
    else:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x,
                                                               cv2.COLOR_RGB2GRAY).reshape(
            x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    try:
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        print('ValueError for Snow, Exception handling')
        x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(
            snow_layer, k=2)
        return np.clip(x, 0, 1) * 255



def spatter(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(0.62,0.1,0.7,0.7,0.5,0),
         (0.65,0.1,0.8,0.7,0.5,0),
         (0.65,0.3,1,0.69,0.5,0),
         (0.65,0.1,0.7,0.69,0.6,1),
         (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]
    elif scale == 'tin':
        c = [(0.62,0.1,0.7,0.7,0.6,0),
         (0.65,0.1,0.8,0.7,0.6,0),
         (0.65,0.3,1,0.69,0.6,0),
         (0.65,0.1,0.7,0.68,0.6,1),
         (0.65,0.1,0.5,0.67,0.6,1)][severity - 1]
    else:
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x_PIL = x
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)

        if len(x.shape) < 3 or x.shape[2] < 3:
            add_spatter_color = cv2.cvtColor(np.clip(m * color, 0, 1),
                                             cv2.COLOR_BGRA2BGR)
            add_spatter_gray = rgb2gray(add_spatter_color)

            return np.clip(x + add_spatter_gray, 0, 1) * 255

        else:

            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1),
                                cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        x_rgb = np.array(x_PIL.convert('RGB'))

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x_rgb[..., :1]),
                                42 / 255. * np.ones_like(x_rgb[..., :1]),
                                20 / 255. * np.ones_like(x_rgb[..., :1])),
                               axis=2)
        color *= m[..., np.newaxis]
        if len(x.shape) < 3 or x.shape[2] < 3:
            x *= (1 - m)
            return np.clip(x + rgb2gray(color), 0, 1) * 255

        else:
            x *= (1 - m[..., np.newaxis])
            return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [.75, .5, .4, .3, 0.15][severity - 1]
    elif scale == 'tin':
        c = [.4, .3, .2, .1, 0.05][severity - 1]
    else:
        c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [.05, .1, .15, .2, .3][severity - 1]
    elif scale == 'tin':
        c = [.1, .2, .3, .4, .5][severity - 1]
    else:
        c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    elif scale == 'tin':
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)][severity - 1]
    else:
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.

    gray_scale = False
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.array([x, x, x]).transpose((1, 2, 0))
        gray_scale = True
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    if gray_scale:
        x = x[:, :, 0]

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [80, 65, 58, 50, 40][severity - 1]
    elif scale == 'tin':
        c = [65, 58, 50, 40, 25][severity - 1]
    else:
        c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    gray_scale = False
    if x.mode != 'RGB':
        gray_scale = True
        x = x.convert('RGB')
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    if gray_scale:
        x = x.convert('L')

    return x


def pixelate(x, severity=1, scale='in'):
    if scale == 'cifar':
        c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    elif scale == 'tin':
        c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    else:
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1, scale='in'):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    if scale == 'cifar':
        IMSIZE=32
        c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
            (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
            (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
            (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
            (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][severity - 1]
    elif scale == 'tin':
        IMSIZE = 64
        c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
            (IMSIZE*0.05, IMSIZE*0.3, IMSIZE*0.06),
            (IMSIZE*0.1, IMSIZE*0.08, IMSIZE*0.06),
            (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03),
            (IMSIZE*0.16, IMSIZE*0.03, IMSIZE*0.02)][severity - 1]
    else:
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
            (244 * 2, 244 * 0.08, 244 * 0.2),
            (244 * 0.05, 244 * 0.01, 244 * 0.02),
            (244 * 0.07, 244 * 0.01, 244 * 0.02),
            (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    
    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

# /////////////// End Corruptions ///////////////


class Transform(abc.ABC):

    name = "abstract_transform"

    def __init__(self, severity, im_size, record=False, max_intensity=False, **kwargs):
        self.im_size = im_size
        self.severity = severity
        self.record = record
        self.max_intensity = max_intensity

    @abc.abstractmethod
    def transform(self, image, **kwargs):
        ...

    @abc.abstractmethod
    def sample_parameters(self):
        ...

    def __call__(self, image):
        params = self.sample_parameters()
        out = self.transform(image, **params)
        if self.record:
            return out, params
        return out

    def is_iterable(obj):
        try:
            iter(obj)
        except:
            return False
        else:
            return True

    def convert_to_numpy(self, params):
        out = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                out.extend(v.flatten().tolist())
            elif self.is_iterable(v):
                out.extend([x for x in v])
            else:
                out.append(v)
        return np.array(out)

    def convert_from_numpy(self, numpy_record):
        param_signature = self.sample_parameters()
        #assert len(param_signature.keys())<=len(numpy_record), "Mismatched numpy_record."
        offset = 0
        for k, v in param_signature.items():
            if isinstance(v, np.ndarray):
                num = len(v.flatten())
                data = numpy_record[offset:offset+num]
                if v.dtype==np.int or v.dtype==np.uint:
                    data = np.round(data, 3)
                data = data.astype(v.dtype)
                param_signature[k] = data.reshape(v.shape)
                offset += num
            elif self.is_iterable(v):
                data = []
                for x in v:
                    if type(x) == 'int':
                        data.append(int(np.round(numpy_record[offset],3)))
                    else:
                        data.append(type(x)(numpy_record[offset]))
                    offset += 1
                param_signature[k] = data
            else:
                if type(v) == 'int':
                    param_signature[k] = int(np.round(numpy_record[offset],3))
                else:
                    param_signature[k] = type(v)(numpy_record[offset])
                offset += 1
        return param_signature

def smoothstep(low, high, x):
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return np.clip(3 * (x ** 2) - 2 * (x ** 3), 0, 1)


def bilinear_interpolation(image, point):
    l = int(np.floor(point[0]))
    u = int(np.floor(point[1]))
    r, d = l+1, u+1
    lu = image[l,u,:] if l >= 0 and l < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    ld = image[l,d,:] if l >= 0 and l < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    ru = image[r,u,:] if r >= 0 and r < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    rd = image[r,d,:] if r >= 0 and r < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    al = lu * (1.0 - point[1] + u) + ld * (1.0 - d + point[1])
    ar = ru * (1.0 - point[1] + u) + rd * (1.0 - d + point[1])
    out = al * (1.0 - point[0] + l) + ar * (1.0 - r + point[0])
    return out

def int_parameter(level, maxval):
  return int(level * maxval / 10)

def float_parameter(level, maxval):
  return float(level) * maxval / 10.

class PerlinNoiseGenerator(object):
    def __init__(self, random_state=None):
        self.rand = np.random if random_state is None else random_state

        B = 256
        N = 16*256

        def normalize(arr):
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else np.zeros_like(arr)

        self.p = np.arange(2*B+2)
        self.g = np.array([normalize((random_state.randint(low=0, high=2**31, size=2) % (2*B) - B )/ B)\
                for i in range(2*B+2)])


        for i in np.arange(B-1,-1,-1):
            k = self.p[i]
            j = self.rand.randint(low=0, high=2**31) % B
            self.p[i] = self.p[j]
            self.p[j] = k

        for i in range(B+2):
            self.p[B+i] = self.p[i]
            self.g[B+i,:] = self.g[i,:]
        self.B = B
        self.N = N


    def s_curve(t):
        return t**2 * (3.0 - 2.0 * t)

    def noise(self, x, y):

        t = x + self.N
        bx0 = int(t) % self.B
        bx1 = (bx0+1) % self.B
        rx0 = t % 1
        rx1 = rx0 - 1.0

        t = y + self.N
        by0 = int(t) % self.B
        by1 = (by0+1) % self.B
        ry0 = t % 1
        ry1 = ry0 - 1.0

        i = self.p[bx0]
        j = self.p[bx1]

        b00 = self.p[i + by0]
        b10 = self.p[j + by0]
        b01 = self.p[i + by1]
        b11 = self.p[j + by1]

        sx = PerlinNoiseGenerator.s_curve(rx0)
        sy = PerlinNoiseGenerator.s_curve(ry0)

        u = rx0 * self.g[b00,0] + ry0 * self.g[b00,1]
        v = rx1 * self.g[b10,0] + ry0 * self.g[b10,1]
        a = u + sx * (v - u)

        u = rx0 * self.g[b01,0] + ry1 * self.g[b01,1]
        v = rx1 * self.g[b11,0] + ry1 * self.g[b11,1]
        b = u + sx * (v - u)

        return 1.5 * (a + sy * (b - a))

    def turbulence(self, x, y, octaves):
        t = 0.0
        f = 1.0
        while f <= octaves:
            t += np.abs(self.noise(f*x, f*y)) / f
            f = f * 2
        return t

class SingleFrequencyGreyscale(Transform):

    name = 'single_frequency_greyscale'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        freq_mag = np.random.uniform(low=-np.pi, high=np.pi)
        freq_2 = np.random.uniform(low=-abs(freq_mag), high=abs(freq_mag))
        freq = np.array([freq_mag, freq_2])[np.random.permutation(2)]
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = float_parameter(self.severity, 196)
        return { 'freq' : freq, 'phase' : phase, 'intensity' : intensity}

    def transform(self, image, freq, phase, intensity):
        noise = np.array([[np.sin(x * freq[0] + y * freq[1] + phase)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((noise, noise, noise), axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['freq'].tolist() + [params['phase'], params['intensity']])

    def convert_from_numpy(self, numpy_record):
        return {'freq' : numpy_record[0:2],
                'phase' : numpy_record[2],
                'intensity' : numpy_record[3]
                }

class CocentricSineWaves(Transform):

    name = 'cocentric_sine_waves'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        offset = np.random.uniform(low=0, high=self.im_size, size=2)
        freq = np.random.uniform(low=0, high=10)
        amplitude = np.random.uniform(low=0, high=self.im_size/10)
        ring_width = np.random.uniform(low=0, high=self.im_size/10)
        intensity = [float_parameter(self.severity, 128) for i in range(3)]

        return { 'offset' : offset,
                 'freq' : freq,
                 'amplitude' : amplitude,
                 'ring_width' : ring_width,
                 'intensity' : intensity
                }

    def transform(self, image, offset, freq, amplitude, ring_width, intensity):

        def calc_intensity(x, y, x0, y0, freq, amplitude, ring_width):
            angle = np.arctan2(x-x0, y-y0) * freq
            distance = ((np.sqrt((x-x0)**2 + (y-y0)**2) + np.sin(angle) * amplitude) % ring_width) / ring_width
            distance -= 1/2
            return distance

        noise = np.array([[calc_intensity(x, y, offset[0], offset[1], freq, amplitude, ring_width)\
                    for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((intensity[0] * noise, intensity[1] * noise, intensity[2] * noise), axis=2)

        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['offset'].tolist() + [params['freq'], params['amplitude'], params['ring_width']] + params['intensity'])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0:2].tolist(),
                'freq' : numpy_record[2],
                'amplitude' : numpy_record[3],
                'ring_width' : numpy_record[4],
                'intensity' : numpy_record[4:7].tolist()
                }
        

class PlasmaNoise(Transform):

    name = 'plasma_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.0, high=6*np.pi)
        iterations = np.random.randint(low=4, high=7)
        sharpness = np.random.uniform(low=0.5, high=1.0)
        scale = np.random.uniform(low=0.075, high=0.2) * self.im_size
        intensity = float_parameter(self.severity,64)
        return {'time' : time, 'iterations' : iterations, 'sharpness' : sharpness,
                'scale' : scale, 'intensity' : intensity}

    def transform(self, image, time, iterations, sharpness, scale, intensity):

        def kernel(x, y, rand, iters, sharp, scale):
            x /= scale
            y /= scale
            i = np.array([1.0, 1.0, 1.0, 0.0])
            for s in range(iters):
                r = np.array([np.cos(y * i[0] - i[3] + rand / i[1]), np.sin(x * i[0] - i[3] + rand / i[1])]) / i[2]
                r += np.array([-r[1],r[0]]) * 0.3
                x += r[0]
                y += r[1]
                i *= np.array([1.93, 1.15, (2.25 - sharp), rand * i[1]])
            r = np.sin(x - rand)
            b = np.sin(y + rand)
            g = np.sin((x + y + np.sin(rand))*0.5)
            return [r,g,b]


        noise = np.array([[kernel(x,y, time, iterations, sharpness, scale)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip((1-intensity/255) * image + intensity * noise, 0, 255).astype(np.uint8)

class CausticNoise(Transform):

    name = 'caustic_noise'
    tags = ['new_corruption']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        intensity = float_parameter(self.severity, 255)

        return { 'time' : time, 'size' : size, 'intensity' : intensity}

    def transform(self, image, time, size, intensity):

        def kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])

        noise = np.array([[kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(image + intensity  *  noise, 0, 255).astype(np.uint8)

class Sparkles(Transform):

    name = 'sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        centers = np.random.uniform(low=0, high=self.im_size, size=(5, 2))
        radii = np.array([float_parameter(self.severity, 0.1)\
                for i in range(5)]) * self.im_size
        amounts = np.array([50 for i in range(5)])
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**31)
        nrays = np.random.randint(low=50, high=200, size=5)

        return {'centers' : centers, 'radii' : radii, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'nrays' : nrays, 'amounts' : amounts
                }

    def transform(self, image, centers, radii, nrays, amounts, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return value + f * (color - value)

        random_state = np.random.RandomState(seed=seed)
        for center, rays, amount, radius in zip(centers, nrays, amounts, radii):
            ray_lengths = [max(1,radius + randomness / 100.0 * radius * random_state.randn())\
                for i in range(rays)]

            image = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)


class InverseSparkles(Transform):

    name = 'inverse_sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = 0.25 * self.im_size
        amount = 100
        amount = float_parameter(self.severity, 65)
        amount = 100 - amount
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**31)
        rays = np.random.randint(low=50, high=200)

        return {'center' : center, 'radius' : radius, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'rays' : rays, 'amount' : amount
                }

    def transform(self, image, center, radius, rays, amount, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return color + f * (value - color)

        random_state = np.random.RandomState(seed=seed)
        ray_lengths = [radius + randomness / 100.0 * radius * random_state.randn()\
                for i in range(rays)]

        out = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(out, 0, 255).astype(np.uint8)

class PerlinNoise(Transform):

    name = 'perlin_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        m = np.array([[1,0],[0,1]]) / (32 * self.im_size / 224)
        turbulence = 16.0
        gain = 0.5
        bias = 0.5
        alpha = float_parameter(self.severity, 0.50)
        seed = np.random.randint(low=0, high=2**31)
        return {'m': m, 'turbulence' : turbulence, 'seed': seed,
                'gain': gain, 'bias': bias, 'alpha': alpha}

    def transform(self, image, m, turbulence, seed, gain, bias, alpha):
        
        random_state = np.random.RandomState(seed=seed)
        noise = PerlinNoiseGenerator(random_state)

        def kernel(point, m, turbulence, gain, bias):
            npoint = np.matmul(point, m)
            f = noise.turbulence(npoint[0], npoint[1], turbulence)\
                    if turbulence != 1.0 else noise.noise(npoint[0], npoint[1])
            f = gain * f + bias
            return np.clip(np.array([f,f,f]),0,1.0)

        noise = np.array([[kernel(np.array([y,x]),m,turbulence,gain, bias) for x in range(self.im_size)]\
                for y in range(self.im_size)])
        out = (1 - alpha) * image.astype(np.float32) + 255 * alpha * noise
        return np.clip(out, 0, 255).astype(np.uint8)

class BlueNoise(Transform):

    name = 'blue_noise'
    tags = ['new_corruption']


    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**31)
        intensity = float_parameter(self.severity, 196)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class BrownishNoise(Transform):

    name = 'brownish_noise'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**31)
        intensity = float_parameter(self.severity, 64)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[1/(np.linalg.norm(np.array([x,y])-center)**2+1)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)
 
class CheckerBoardCutOut(Transform):

    name = 'checkerboard_cutout'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        scales = np.maximum(np.random.uniform(low=0.1, high=0.25) * self.im_size, 1)
        scales = (scales, scales)
        fraction = float_parameter(self.severity, 1.0)
        seed = np.random.randint(low=0, high=2**31)

        return {'angle' : angle, 'scales' : scales, 'fraction' : fraction, 'seed' : seed}

    def transform(self, image, scales, angle, fraction, seed):
        random_state = np.random.RandomState(seed=seed)
        grid = random_state.uniform(size=(int(4*self.im_size//scales[0]), int(4*self.im_size//scales[1]))) < fraction
        
        def mask_kernel(point, scales, angle, grid):
            nx = (np.cos(angle) * point[0] + np.sin(angle) * point[1]) / scales[0]
            ny = (-np.sin(angle) * point[0] + np.cos(angle) * point[1]) / scales[1]
            return (int(nx % 2) != int(ny % 2)) or not grid[int(nx),int(ny)]

        out = np.array([[image[y,x,:] if mask_kernel([y,x], scales, angle, grid) else np.array([128,128,128])\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(out, 0, 255).astype(np.uint8)


class Lines(Transform):

    name = 'lines'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        length = 1.0
        density = float_parameter(self.severity, 1.0)
        angle = np.random.uniform(low=0.0, high=2*np.pi)
        angle_variation = np.random.uniform(low=0.1, high=1.0)
        seed = np.random.randint(low=0, high=2**31)

        return {'length' : length, 'density' : density, 'angle' : angle, 'angle_variation' : angle_variation, 'seed' : seed}

    def transform(self, image, length, density, angle, angle_variation, seed):

        num_lines = int(density * self.im_size)
        l = length * self.im_size
        random_state = np.random.RandomState(seed=seed)
        out = image.copy()
        for i in range(num_lines):
            x = self.im_size * random_state.uniform()
            y = self.im_size * random_state.uniform()
            a = angle + 2 * np.pi * angle_variation * (random_state.uniform() - 0.5)
            s = np.sin(a) * l
            c = np.cos(a) * l
            x1 = int(x-c)
            x2 = int(x+c)
            y1 = int(y-s)
            y2 = int(y+s)
            rxc, ryc, rval = line_aa(x1, y1, x2, y2)
            xc, yc, val = [], [], []
            for rx, ry, rv in zip(rxc, ryc, rval):
                if rx >= 0 and ry >= 0 and rx < self.im_size and ry < self.im_size:
                    xc.append(rx)
                    yc.append(ry)
                    val.append(rv)
            xc, yc, val = np.array(xc, dtype=np.int32), np.array(yc, dtype=np.int32), np.array(val)
            out[xc, yc, :] = (1.0 - val.reshape(-1,1)) * out[xc, yc, :].astype(np.float32) + val.reshape(-1,1)*128
        return out.astype(np.uint8)


class BlueNoiseSample(Transform):

    name = 'blue_noise_sample'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**31)
        threshold = float_parameter(self.severity, 3.0) - 2.5

        return {'seed' : seed, 'threshold' : threshold}

    def transform(self, image, seed, threshold):
        random_state = np.random.RandomState(seed=seed)

        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        mask = noise > threshold
        out = image * mask.reshape(self.im_size, self.im_size, 1)

        return np.clip(out, 0, 255).astype(np.uint8)

class CausticRefraction(Transform):

    name = 'caustic_refraction'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        eta = 4.0
        lens_scale = float_parameter(self.severity, 0.5*self.im_size)
        lighting_amount = float_parameter(self.severity, 2.0)
        softening = 1

        return { 'time' : time, 'size' : size, 'eta' : eta, 'lens_scale' : lens_scale, 'lighting_amount': lighting_amount, 'softening' : softening}

    def transform(self, image, time, size, eta, lens_scale, lighting_amount, softening):

        def caustic_noise_kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])


        def refract(incident, normal, eta):
            if np.abs(np.dot(incident, normal)) >= 1.0 - 1e-3:
                return incident
            angle = np.arccos(np.dot(incident, normal))
            out_angle = np.arcsin(np.sin(angle) / eta)
            out_unrotated = np.array([np.cos(out_angle), np.sin(out_angle), 0.0])
            spectator_dim = np.cross(incident, normal)
            spectator_dim /= np.linalg.norm(spectator_dim)
            orthogonal_dim = np.cross(normal, spectator_dim)
            rotation_matrix = np.stack((normal, orthogonal_dim, spectator_dim), axis=0)
            return np.matmul(np.linalg.inv(rotation_matrix), out_unrotated)

        def luma_at_offset(image, origin, offset):
            pixel_value = image[origin[0]+offset[0], origin[1]+offset[1], :]\
                    if origin[0]+offset[0] >= 0 and origin[0]+offset[0] < image.shape[0]\
                    and origin[1]+offset[1] >= 0 and origin[1]+offset[1] < image.shape[1]\
                    else np.array([0.0,0.0,0])
            return np.dot(pixel_value, np.array([0.2126, 0.7152, 0.0722]))

        def luma_based_refract(point, image, caustics, eta, lens_scale, lighting_amount):
            north_luma = luma_at_offset(caustics, point, np.array([0,-1]))
            south_luma = luma_at_offset(caustics, point, np.array([0, 1]))
            west_luma = luma_at_offset(caustics, point, np.array([-1, 0]))
            east_luma = luma_at_offset(caustics, point, np.array([1,0]))

            lens_normal = np.array([east_luma - west_luma, south_luma - north_luma, 1.0])
            lens_normal = lens_normal / np.linalg.norm(lens_normal)

            refract_vector = refract(np.array([0.0, 0.0, 1.0]), lens_normal, eta) * lens_scale
            refract_vector = np.round(refract_vector, 3)

            out_pixel = bilinear_interpolation(image, point+refract_vector[0:2])
            out_pixel += (north_luma - south_luma) * lighting_amount
            out_pixel += (east_luma - west_luma) * lighting_amount

            return np.clip(out_pixel, 0, 1)

        noise = np.array([[caustic_noise_kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = gaussian_filter(noise, sigma=softening)

        image = image.astype(np.float32) / 255
        out = np.array([[luma_based_refract(np.array([y,x]), image, noise, eta, lens_scale, lighting_amount)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip((out * 255).astype(np.uint8), 0, 255)

class PinchAndTwirl(Transform):

    name = 'pinch_and_twirl'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        num_per_axis = 5 if self.im_size==224 else 3
        angles = np.array([np.random.choice([1,-1]) * float_parameter(self.severity, np.pi/2) for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)

        amount = float_parameter(self.severity, 0.4) + 0.1
        return {'num_per_axis' : num_per_axis, 'angles' : angles, 'amount' : amount}

    def transform(self, image, num_per_axis, angles, amount):

        def warp_kernel(point, center, radius, amount, angle):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dist = np.linalg.norm(point - center)

            if dist > radius or np.round(dist, 3) == 0.0:
                return point

            d = dist / radius
            t = np.sin(np.pi * 0.5 * d) ** (- amount)

            dx *= t
            dy *= t

            e = 1 - d
            a = angle * (e ** 2)
            
            out = center + np.array([dx*np.cos(a) - dy*np.sin(a), dx*np.sin(a) + dy*np.cos(a)])

            return out

        out = image.copy().astype(np.float32)
        grid_size = self.im_size // num_per_axis
        radius = grid_size / 2
        for i in range(num_per_axis):
            for j in range(num_per_axis):
                l, r = i * grid_size, (i+1) * grid_size
                u, d = j * grid_size, (j+1) * grid_size
                center = np.array([u+radius, l+radius])
                out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, amount, angles[i,j]))\
                        for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)

class Ripple(Transform):

    name = 'ripple'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amplitudes = np.array([float_parameter(self.severity, 0.025)\
                for i in range(2)]) * self.im_size
        wavelengths = np.random.uniform(low=0.1, high=0.3, size=2) * self.im_size
        phases = np.random.uniform(low=0, high=2*np.pi, size=2)
        return {'amplitudes' : amplitudes, 'wavelengths' : wavelengths, 'phases' : phases}

    def transform(self, image, wavelengths, phases, amplitudes):

        def warp_kernel(point, wavelengths, phases, amplitudes):
            return point + amplitudes * np.sin(2 * np.pi * point / wavelengths + phases)

        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), wavelengths, phases, amplitudes))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8) 


class TransverseChromaticAbberation(Transform):

    name = 'transverse_chromatic_abberation'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        scales = np.array([float_parameter(self.severity, 0.5)\
                for i in range(3)])
        scale = float_parameter(self.severity, 0.5)
        scales = np.array([1.0, 1.0+scale/2, 1.0+scale])
        scales = scales[np.random.permutation(3)]

        return { 'scales' : scales }

    def transform(self, image, scales):
        out = image.copy()
        for c in range(3):
            zoomed = zoom(image[:,:,c], scales[c], prefilter=False)
            edge = (zoomed.shape[0]-self.im_size)//2
            out[:,:,c] = zoomed[edge:edge+self.im_size, edge:edge+self.im_size]
        return out.astype(np.uint8)
            
    def convert_to_numpy(self, params):
        return params['scales'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'scales' : numpy_record}


class CircularMotionBlur(Transform):

    name = 'circular_motion_blur'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amount = float_parameter(self.severity,15)

        return {'amount' : amount}

    def transform(self, image, amount):

        num = 21
        factors = []
        rotated = []
        image = image.astype(np.float32) / 255
        for i in range(num):
            angle = (2*i/(num-1) - 1) * amount
            rotated.append(rotate(image, angle, reshape=False))
            factors.append(np.exp(- 2*(2*i/(num-1)-1)**2))
        out = np.zeros_like(image)
        for i, f in zip(rotated, factors):
            out += f * i
        out /= sum(factors)
        return np.clip(out*255, 0, 255).astype(np.uint8)
    