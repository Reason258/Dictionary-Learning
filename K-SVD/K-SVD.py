from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.metrics import structural_similarity as ssim

import math

# Define the PSNR function
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Define the SSIM function
stdVec = [5, 10, 15, 25]
for item in stdVec:
    # Set the std [5 10 15 25]
    std = item

    # load an image from file
    image = Image.open('296059.jpg')

    # convert the image pixels to a numpy array
    image = np.array(image)
    image = image[:,:,0]

    #print(image)
    print("original shape", image.shape)
    image = image.astype('float32')
    image/=255
    plt.imshow(image, cmap='gray')
    #plt.show()

    # Add noise to the image
    noise = np.random.normal(loc=0, scale=std, size=image.shape)/255
    x_test_noisy1 = image + noise
    x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)
    plt.imshow(x_test_noisy1, cmap='Greys_r')
    #plt.show()

    # Calculate the PSNR and SSIM
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
    dico = MiniBatchDictionaryLearning(n_components=144, alpha=1, n_iter=500)
    V = dico.fit(data).components_
    print(V.shape)

    # Display the dictionary
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:144]):

        plt.subplot(12, 12, i + 1)

        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,

        interpolation='nearest')

        plt.xticks(())

        plt.yticks(())

        plt.suptitle('Dictionary learned from patches\n', fontsize=16)

        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    #plt.show()


    # Encapsulate the testing part
    def test(filename):

        print('Test the dictionary on a new image')
        # load an image from file
        image = Image.open(filename)

        # convert the image pixels to a numpy array
        image = np.array(image)
        image = image[:,:,0]

        #print image
        print("original shape", image.shape)
        image = image.astype('float32')
        image/=255
        plt.imshow(image, cmap='gray')
        #plt.show()

        # Add noise to the image
        noise = np.random.normal(loc=0, scale=std, size=image.shape)/255
        x_test_noisy1 = image + noise
        #print x_test_noisy1
        x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)
        psnr1 = psnr(image, x_test_noisy1)
        ssim1 = ssim(image, x_test_noisy1)
        plt.imshow(x_test_noisy1, cmap='Greys_r')
        #plt.show()

        # Extract noisy patches and reconstruct them using the dictionary
        print('Extracting patches from new image... ')
        #print x_test_noisy1
        data = extract_patches_2d(x_test_noisy1, patch_size)
        data = data.reshape(data.shape[0], -1)
        intercept = np.mean(data, axis=0)
        data -= intercept

        #  Reconstruct the image using the dictionary
        print('Orthogonal Matching Pursuit\n2 atoms' + '...')
        reconstructions_frm_noise = x_test_noisy1.copy()
        dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
        code = dico.transform(data)
        patches = np.dot(code, V)
        patches += intercept
        patches = patches.reshape(len(data), *patch_size)
        reconstructions_frm_noise = reconstruct_from_patches_2d(patches, image.shape)

        # Calculate the PSNR and SSIM
        psnr2 = psnr(image, reconstructions_frm_noise)
        ssim2 = ssim(image, reconstructions_frm_noise)

        # Display the results
        print('std: {}'.format(std))
        print('{} PSNR_noise: '.format(filename), psnr1)
        print('{} PSNR_denoise: '.format(filename), psnr2)
        print('{} SSIM_noise: '.format(filename), ssim1*100)
        print('{} SSIM_denoise: '.format(filename), ssim2*100)
        plt.imshow(reconstructions_frm_noise, cmap='Greys_r')
        #plt.show()

    test('296059.jpg')
    test('100080.jpg')
    test('12003.jpg')

    #Modified test function to return metrics
def test(filename, std, V, patch_size):
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0, scale=std, size=image.shape) / 255
    x_test_noisy = np.clip(image + noise, 0., 1.)
    
    psnr_noisy = psnr(image, x_test_noisy)
    ssim_noisy = ssim(image, x_test_noisy)

    #Extracting patches from noisy image
    data = extract_patches_2d(x_test_noisy, patch_size).reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept

    #Using dictionary to denoise
    dico = MiniBatchDictionaryLearning(n_components=144, alpha=1, n_iter=500, transform_algorithm='omp', transform_n_nonzero_coefs=2)
    dico.components_ = V
    code = dico.transform(data)
    patches = np.dot(code, V)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    reconstructed = reconstruct_from_patches_2d(patches, image.shape)
    
    psnr_denoised = psnr(image, reconstructed)
    ssim_denoised = ssim(image, reconstructed)
    
    return psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised

stdVec = [5, 10, 15, 25]
images = ['296059.jpg', '100080.jpg', '12003.jpg']
results = {img: {std: {} for std in stdVec} for img in images}

patch_size = (8, 8)

# Assuming 'V' is obtained from some prior dictionary learning step
# Placeholder for 'V' - actual dictionary learning to be performed as needed
V = np.random.rand(144, 64)  # Placeholder dictionary

for std in stdVec:
    for img in images:
        psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised = test(img, std, V, patch_size)
        results[img][std] = {
            'PSNR_Noisy': psnr_noisy,
            'PSNR_Denoised': psnr_denoised,
            'SSIM_Noisy': ssim_noisy,
            'SSIM_Denoised': ssim_denoised
        }

#Plotting the results
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
for img in images:
    axs[0].plot(stdVec, [results[img][std]['PSNR_Noisy'] for std in stdVec], '-o', label=f'{img} PSNR_Noisy')
    axs[0].plot(stdVec, [results[img][std]['PSNR_Denoised'] for std in stdVec], '--o', label=f'{img} PSNR_Denoised')
    axs[1].plot(stdVec, [results[img][std]['SSIM_Noisy'] for std in stdVec], '-o', label=f'{img} SSIM_Noisy')
    axs[1].plot(stdVec, [results[img][std]['SSIM_Denoised'] for std in stdVec], '--o', label=f'{img} SSIM_Denoised')

axs[0].set_xlabel('Noise STD')
axs[0].set_ylabel('PSNR')
axs[0].legend()
axs[0].set_title('PSNR vs Noise STD')

axs[1].set_xlabel('Noise STD')
axs[1].set_ylabel('SSIM')
axs[1].legend()
axs[1].set_title('SSIM vs Noise STD')

plt.tight_layout()
plt.show()

