import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import random
import numpy as np
import cv2
import torch.fft as fft

class DenoisingUSTestDataset(Dataset):
    def __init__(self, original_dir, filtered_dir, phase='train', target_filter='beta_10_H_5_', test_noise_level=0,\
                 test_blur_level=0, augmented=True, test_noise_mode='gaussian'):
        self.original_dir = original_dir
        self.filtered_dir = filtered_dir
        self.target_filter = target_filter
        self.phase = phase
        self.augmented = augmented
        self.image_filenames = [f for f in os.listdir(original_dir) if f.endswith('.png')]
        
        if self.phase == 'train':
            self.image_filenames = self.image_filenames[33:]
        elif self.phase == 'val':
            self.image_filenames = self.image_filenames[16:33]
        else:
            self.image_filenames = self.image_filenames[0:16]

        # Define augmentation transforms for training
        if self.phase == 'train':
            self.resize = transforms.Resize((128, 128))  # Resize to a larger size before cropping
            self.to_tensor = transforms.ToTensor()
            self.grayscale = transforms.Grayscale()
        elif self.phase == 'test':
            self.resize = transforms.Resize((128, 128))  # Resize to a larger size before cropping
            self.to_tensor = transforms.ToTensor()
            self.grayscale = transforms.Grayscale()
            self.test_noise_mode = test_noise_mode
            self.test_noise_level = test_noise_level
            self.test_blur_level = test_blur_level
        else:
            # No augmentation for validation/test phase
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, self.image_filenames[idx])
        filtered_image_path = os.path.join(self.filtered_dir,  self.target_filter + self.image_filenames[idx])

        original_image = Image.open(original_image_path)
        filtered_image = Image.open(filtered_image_path)

        # Added on May 04, 2025
        original_image_tensor = self.to_tensor(self.grayscale(self.resize(original_image)))
        if self.phase == 'train':
            # Apply same transformation to both images
            degraded_image, filtered_image = self.apply_transforms(original_image, filtered_image)
            return original_image_tensor, degraded_image, filtered_image
        elif self.phase == 'test':
            # Apply same transformation to both images
            degraded_image, filtered_image = self.apply_transforms_test(original_image, filtered_image)
            # Apply padding for validation and test phase
            degraded_image, pad_sizes = self.mirror_padding(degraded_image)
            return original_image_tensor, degraded_image, filtered_image, pad_sizes
        else:
            # For validation, apply padding and other transformations
            degraded_image = self.transform(original_image)
            filtered_image = self.transform(filtered_image)
            # print('original_image:',original_image.shape,'filtered_image:',filtered_image.shape)
            # Apply padding for validation and test phase
            degraded_image, pad_sizes = self.mirror_padding(degraded_image)
            # print('After_pad original_image:',original_image.shape,'filtered_image:',filtered_image.shape)
            return original_image_tensor, degraded_image, filtered_image, pad_sizes


    def apply_transforms_test(self, original_image, filtered_image):
        """ Apply the same transformations to both original and target images """
        # Resize both images to the same size
        original_image = self.resize(original_image)
        filtered_image = self.resize(filtered_image)

        # Convert to grayscale and tensor
        original_image = self.to_tensor(self.grayscale(original_image))
        filtered_image = self.to_tensor(self.grayscale(filtered_image))

        # Randomly apply noise and blur to the original image
        blur_strength = self.test_blur_level
        noise_mode = self.test_noise_mode
        noise_strength = self.test_noise_level
        # if(noise_strength>0):
            # print('Adding noise')
        if(noise_mode == 'gaussian'):
            original_image = self.add_fourier_complex_gaussian_noise(original_image, noise_strength, self.phase)
        else:
            original_image = self.add_speckle_noise(original_image, noise_strength, self.phase)
        
        if(blur_strength>0):
            # print('Applying blur')
            original_image = self.apply_gaussian_blur(original_image, blur_strength)
        
        return original_image, filtered_image

    def apply_transforms(self, original_image, filtered_image):
        """ Apply the same transformations to both original and target images """
        
        # Resize both images to a larger size
        original_image = self.resize(original_image)
        filtered_image = self.resize(filtered_image)
        
        if self.augmented == True:
            # Apply the same random rotation to both original and filtered images
            rotation_angle = transforms.RandomRotation.get_params([-15, 15])  # Rotation between -15 to 15 degrees
            original_image = transforms.functional.rotate(original_image, rotation_angle)
            filtered_image = transforms.functional.rotate(filtered_image, rotation_angle)
        
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(original_image, output_size=(64, 64))
            original_image = transforms.functional.crop(original_image, i, j, h, w)
            filtered_image = transforms.functional.crop(filtered_image, i, j, h, w)
        else:
            # Center crop
            original_image = transforms.functional.center_crop(original_image, output_size=(128,128))
            filtered_image = transforms.functional.center_crop(filtered_image, output_size=(128,128))
            # i, j, h, w = transforms.CenterCrop.get_params(original_image, output_size=(64, 64))
            # original_image = transforms.functional.crop(original_image, i, j, h, w)
            # filtered_image = transforms.functional.crop(filtered_image, i, j, h, w)            

        # Convert to grayscale and tensor
        original_image = self.to_tensor(self.grayscale(original_image))
        filtered_image = self.to_tensor(self.grayscale(filtered_image))

        # Randomly apply noise and blur to the original image
        noise_type = random.choice([self.add_noise, self.add_fourier_complex_gaussian_noise])
        if random.random() > 0.55:
            # original_image = noise_type(original_image)
            blur_strength = random.choice([3, 5, 7, 9, 11, 13, 15, 17])  # Random blur kernel size (3x3, 5x5, or 7x7)
            original_image = self.apply_gaussian_blur(original_image,blur_strength)
            original_image = noise_type(original_image)
        if random.random() < 0.45:
            blur_strength = 3
            original_image = self.add_fourier_complex_gaussian_noise(original_image)
            original_image = self.apply_gaussian_blur(original_image,blur_strength)
        
        return original_image, filtered_image

    def add_noise(self, img_tensor):
        """ Adds Gaussian noise to the input image with random noise level """
        noise_level = random.uniform(0.05, 0.2)  # Random noise level between 0.05 and 0.2
        noise = torch.randn(img_tensor.size()) * noise_level
        noisy_img = img_tensor + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)  # Keep pixel values in [0,1]
        return noisy_img

    def add_fourier_complex_gaussian_noise(self, img_tensor, noise_strength=0, phase='train'):
        """ Adds complex Gaussian noise in the Fourier domain to the input image """
        img_np = img_tensor.squeeze(0).numpy()  # Convert to HxW format for grayscale

        # Step 1: Apply 2D FFT to the image
        fft_img = np.fft.fft2(img_np)

        # Step 2: Create complex Gaussian noise
        noise_real = np.random.normal(0, 0.1, fft_img.shape)
        noise_imag = np.random.normal(0, 0.1, fft_img.shape)
        complex_noise = noise_real + 1j * noise_imag

        # Step 3: Add the complex noise to the FFT of the image
        if(phase=='test'):
            noisy_fft_img = (1-noise_strength)*fft_img + np.max(fft_img)*complex_noise*noise_strength
        else:
            noise_strength = np.random.uniform(0,0.2)
            noisy_fft_img = (1-noise_strength)*fft_img + np.max(fft_img)*complex_noise*noise_strength

        # Step 4: Apply inverse 2D FFT to get the noisy image back in the spatial domain
        noisy_img_spatial = np.fft.ifft2(noisy_fft_img)

        # Step 5: Take the magnitude (absolute value) of the noisy image
        noisy_img_magnitude = np.abs(noisy_img_spatial)
        
        # Convert back to tensor format
        noisy_img_tensor = torch.from_numpy(noisy_img_magnitude).unsqueeze(0)  # Add channel dimension back
        noisy_img_tensor = torch.clamp(noisy_img_tensor, 0.0, 1.0)  # Keep pixel values in [0,1]

        return noisy_img_tensor

    def add_speckle_noise(self, img_tensor, noise_strength=0, phase='train'):
        """ Adds Gamma distributed Speckle noise to the input image """
        img_np = img_tensor.squeeze(0).numpy()  # Convert to HxW format for grayscale
        if(phase=='test'):
            noise = np.random.gamma(shape=noise_strength, scale=1/noise_strength, 
                                    size=img_np.shape)
            noisy_img = np.exp(np.log(img_np + 1e-3) + np.log(noise + 1e-3))
        else:
            # No noise
            noisy_img = img_np

        # Convert back to tensor format
        noisy_img_tensor = torch.from_numpy(noisy_img).unsqueeze(0)  # Add channel dimension back
        noisy_img_tensor = torch.clamp(noisy_img_tensor, 0.0, 1.0)  # Keep pixel values in [0,1]
        
        return noisy_img_tensor

    def apply_gaussian_blur(self, img_tensor, blur_strength):
        """ Applies Gaussian blur on the image with random blur strength """
        # blur_strength = random.choice([3, 5, 7])  # Random blur kernel size (3x3, 5x5, or 7x7)
        np_img = img_tensor.squeeze(0).numpy()  # Convert to HxW format for grayscale
        blurred = cv2.GaussianBlur(np_img, (blur_strength, blur_strength), 0)
        blurred_tensor = torch.from_numpy(blurred).unsqueeze(0)  # Back to 1xHxW
        return blurred_tensor

    # def mirror_padding(self, img_tensor):
    #     """ Apply mirror padding and return padding sizes """
    #     print('img_tensor.shape:',img_tensor.shape)
    #     h, w = img_tensor.shape[1], img_tensor.shape[2]
    #     pad_h = (64 - h % 64) // 2 if h % 64 != 0 else 0
    #     pad_w = (64 - w % 64) // 2 if w % 64 != 0 else 0
        
    #     padding = (pad_w, pad_w, pad_h, pad_h)
    #     padded_img = torch.nn.functional.pad(img_tensor, padding, mode='reflect')
        
    #     return padded_img, padding
    
    def mirror_padding(self, img_tensor):
        """Apply mirror padding and return padding sizes."""
        # print('img_tensor.shape:', img_tensor.shape)

        # Extract height and width
        h, w = img_tensor.shape[1], img_tensor.shape[2]

        # Calculate padding needed to make dimensions divisible by 64
        pad_h = (64 - h % 64) if h % 64 != 0 else 0
        pad_w = (64 - w % 64) if w % 64 != 0 else 0

        # Split the padding equally on both sides (extra goes to the right/bottom)
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # Create padding tuple: (left, right, top, bottom)
        padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)

        # Apply mirror padding using reflect mode
        padded_img = torch.nn.functional.pad(img_tensor, padding, mode='reflect')

        return padded_img, padding



def custom_collate_fn(batch):
    """Pads images in the batch to the size of the largest image."""
    
    if len(batch[0]) == 2:  # Training Phase (no padding sizes)
        original_images, degraded_images, filtered_images = zip(*batch)
        
        # Find the maximum height and width in this batch
        max_height = max([img.shape[1] for img in original_images])
        max_width = max([img.shape[2] for img in original_images])
        
        # Pad all images to the same size
        padded_original_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in original_images]
        padded_degraded_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in degraded_images]
        padded_filtered_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in filtered_images]
        
        return torch.stack(padded_original_images), torch.stack(padded_degraded_images), torch.stack(padded_filtered_images)

    elif len(batch[0]) == 3:  # Validation/Test Phase (with padding sizes)
        original_images, degraded_images, filtered_images, pad_sizes = zip(*batch)
        
        # Find the maximum height and width in this batch
        max_height = max([img.shape[1] for img in original_images])
        max_width = max([img.shape[2] for img in original_images])
        
        # Pad all images to the same size
        padded_original_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in original_images]
        padded_degraded_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in degraded_images]
        padded_filtered_images = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), mode='constant', value=0) for img in filtered_images]
        
        return torch.stack(padded_original_images), torch.stack(padded_degraded_images), torch.stack(padded_filtered_images), pad_sizes