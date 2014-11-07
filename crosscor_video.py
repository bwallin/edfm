import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tifffile
from progressbar import ProgressBar

import psf
import crosscorr


psf_template_filepath = '/home/bwallin/ws/edf_micro/data/\
Intensity_PSF_template_CirCau_NA003.tif'
psf_template = psf.DiscretizedPSF()
psf_template.load_from_tiff(psf_template_filepath)
tensor = psf_template.to_dense()
r, p, q = tensor.shape

image_filepath = '/home/bwallin/ws/edf_micro/data/\
Camera_image_CirCau_10_dots_NL_10.tif'
image = tifffile.imread(image_filepath)

corr = crosscorr.compute_xcorr_volume_fft(image, tensor)

fig = plt.figure()
progress = ProgressBar()
ims = []
for i in progress(xrange(0, r, 5)):
    ims.append((plt.pcolormesh(corr[i, :, :]), ))

ani = manimation.ArtistAnimation(fig, ims, interval=50,
                                 blit=True, repeat_delay=1000)

ani.save('crosscor_video.mp4', metadata={'artist': 'Bruce'})
