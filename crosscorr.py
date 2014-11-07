import scipy.signal
import scipy.ndimage
from scipy import ones, zeros, sqrt
from scipy.linalg import norm
from progressbar import ProgressBar


def compute_local_norm(image, shape):
    local_norm = sqrt(scipy.ndimage.convolve(image**2, ones(shape)))
    return local_norm


def compute_xcorr_volume(image, tensor):
    n, m = image.shape
    r, p, q = tensor.shape
    corr = zeros((r, n, m))
    local_norm = compute_local_norm(image)

    progress = ProgressBar()
    for k in progress(xrange(0, r)):
        kernel = tensor[k, :, :]
        norm_factors = norm(kernel)*local_norm
        convo = scipy.ndimage.convolve(image, kernel)
        corr[k, :, :] = convo*norm_factors

    return corr


def compute_local_norm_fft(image, shape):
    p, q = shape
    local_norm = sqrt(scipy.signal.fftconvolve(image**2, ones(shape)))
    local_norm = local_norm[p/2:-p/2+1, q/2:-q/2+1]
    return local_norm


def compute_xcorr_volume_fft(image, tensor):
    n, m = image.shape
    r, p, q = tensor.shape
    corr = zeros((r, n, m))
    local_norm = compute_local_norm_fft(image, (p, q))
    progress = ProgressBar()
    for k in progress(xrange(0, r)):
        kernel = tensor[k, :, :]
        norm_factors = norm(kernel)*local_norm
        convo = scipy.signal.fftconvolve(image, kernel)
        convo = convo[p/2:-p/2+1, q/2:-q/2+1]
        corr[k, :, :] = convo/norm_factors

    return corr
