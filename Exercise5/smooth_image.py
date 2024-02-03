import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Reads a square image in 8-bit/color PPM format from the given file. Note: No checks on valid format are done.
def readImage(filename):
    f = open(filename,"rb")
    
    f.readline()
    s = f.readline()
    f.readline()
    (pixel, pixel) = [t(s) for t,s in zip((int,int),s.split())]
    
    data = np.fromfile(f,dtype=np.uint8,count = pixel*pixel*3)
    img = data.reshape((pixel,pixel,3)).astype(np.double)
    
    f.close()
    
    return img, pixel
    

#Writes a square image in 8-bit/color PPM format.
def writeImage(filename, image):
    f = open(filename,"wb")
    
    pixel = image.shape[0]
    header= "P6\n%d %d\n%d\n"%(pixel, pixel, 255)
    f.write(bytearray(header, 'ascii'))
    
    image = image.astype(np.uint8)
    
    image.tofile(f)
    
    f.close()
    
    
# function to compute the power spectrum
def _compute_spectrum_2D(fourier_2D_data):

    npix = fourier_2D_data.shape[0]

    fourier_amplitudes = np.abs(fourier_2D_data)**2
    
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "sum",
                                     bins = kbins)
    Abins = Abins * np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins    



img, pixel = readImage("ngc628-original.ppm")


#Now we set up our desired smoothing kernel. We'll use complex number for it even though it is real. 
kernel_real = np.zeros((pixel,pixel),dtype=np.complex)

hsml = 10.

#now set the values of the kernel 
for i in np.arange(pixel):
    for j in np.arange(pixel):
        
        #TODO: do something sensible here to set the real part of the kernel
        #kernel_real[i, j] = ....
        


#Let's calculate the Fourier transform of the kernel
kernel_kspace = np.fft.fft2(kernel_real)


#further space allocations for image transforms
color_real = np.zeros((pixel,pixel),dtype=np.complex)

# variables for the spectra
# - store spectrum of the original image and the smoothed one
# - three spectra for each color
spectrum_original = np.zeros((3,pixel//2))
spectrum_filter   = np.zeros((3,pixel//2))


#we now convolve each color channel with the kernel using FFTs
for colindex in np.arange(3):
    #copy input color into complex array
    color_real[:,:].real = img[:,:,colindex]
    
    
    #forward transform
    color_kspace = np.fft.fft2(color_real)

    #store image spectrum for original image
    #TODO: compute spectrum and store it in spectrum_original[colindex]
    
    #multiply with kernel in Fourier space
    #TODO: fill in code here
    
    #store image spectrum for smoothed image
    #TODO: compute spectrum and store it in spectrum_filter[colindex]
    
    #backward transform
    color_real = np.fft.ifft2(color_kspace)
    
    #copy real value of complex result back into color array
    img[:,:,colindex] = color_real.real
    
writeImage("ngc628-smoothed.ppm", img)
