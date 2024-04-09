from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.metrics import structural_similarity as ssim
 
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Set the std [5 10 15 25]
std = 15

# load an image from file
image = Image.open('296059.jpg')

# convert the image pixels to a numpy array
image = np.array(image)
image = image[:,:,0]

 
print("original shape", image.shape)
image = image.astype('float32')
image/=255
plt.imshow(image, cmap='gray')
plt.show()

noise = np.random.normal(loc=0, scale=std, size=image.shape)/255
x_test_noisy1 = image + noise
x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)
plt.imshow(x_test_noisy1, cmap='Greys_r')
plt.show()

print('Extracting reference patches...')
patch_size = (8, 8)
data = extract_patches_2d(x_test_noisy1, patch_size)
print(data.shape)
data = data.reshape(data.shape[0], -1)
np.random.shuffle(data)

# Task requirement: uniformly select 6000 blocks (8x8) to learn the dictionary
data = data[0:6000]
print(data.shape)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

# Learn the dictionary from reference patches
print('Learning the dictionary...')
dico = MiniBatchDictionaryLearning(n_components=144, alpha=1, n_iter=1000)
V = dico.fit(data).components_
print(V.shape)

 
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:144]):

    plt.subplot(12, 12, i + 1)

    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,

    interpolation='nearest')

    plt.xticks(())

    plt.yticks(())

    plt.suptitle('Dictionary learned from patches\n', fontsize=16)

    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()


# Encapsulate the testing part
def test(filename):

    print('Test the dictionary on a new image')
    # load an image from file
    image = Image.open(filename)

    # convert the image pixels to a numpy array
    image = np.array(image)
    image = image[:,:,0]

    
    print("original shape", image.shape)
    image = image.astype('float32')
    image/=255
    plt.imshow(image, cmap='gray')
    # plt.show()

    noise = np.random.normal(loc=0, scale=std, size=image.shape)/255
    x_test_noisy1 = image + noise
    
    x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)
    psnr1 = psnr(image, x_test_noisy1)
    ssim1 = ssim(image, x_test_noisy1)
    plt.imshow(x_test_noisy1, cmap='Greys_r')
    # plt.show()

    # Extract noisy patches and reconstruct them using the dictionary
    print('Extracting patches from new image... ')

    data = extract_patches_2d(x_test_noisy1, patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept

    
    print('Orthogonal Matching Pursuit\n2 atoms' + '...')
    reconstructions_frm_noise = x_test_noisy1.copy()
    dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
    code = dico.transform(data)
    patches = np.dot(code, V)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    reconstructions_frm_noise = reconstruct_from_patches_2d(patches, image.shape)

    psnr2 = psnr(image, reconstructions_frm_noise)
    ssim2 = ssim(image, reconstructions_frm_noise)

    print('{} PSNR_noise: '.format(filename), psnr1)
    print('{} PSNR_denoise: '.format(filename), psnr2)
    print('{} SSIM_noise: '.format(filename), ssim1*100)
    print('{} SSIM_denoise: '.format(filename), ssim2*100)
    plt.imshow(reconstructions_frm_noise, cmap='Greys_r')
    # plt.show()

test('296059.jpg')
test('100080.jpg')