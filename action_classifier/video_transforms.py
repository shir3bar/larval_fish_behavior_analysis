import torch
from slowfast.datasets.transform import color_jitter
import cv2
import torchvision.transforms
from pytorchvideo.transforms import Permute
from torchvision.transforms import GaussianBlur

class RandomColorJitter(object):

    def __init__(self, brightness_ratio=0, contrast_ratio=0, saturation_ratio=0, p=0.5):

        self.brightness_ratio = brightness_ratio
        self.contrast_ratio = contrast_ratio
        self.saturation_ratio = saturation_ratio
        self.probability = p


    def __call__(self, x):
        if self.probability < torch.rand(1):
            return x
        else:
            x_mod = color_jitter(x, img_brightness=self.brightness_ratio,
                         img_contrast=self.contrast_ratio,
                         img_saturation=self.saturation_ratio)
            return x_mod
    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.probability})"

class RandomGaussianBlur(object):
    """
    Apply GaussianBlur, specified by kernel and sigma, to each video frame with a certain probability p.
    Video is assumed to be of a shape [C, T, H, W].
    """
    def __init__(self, kernel, sigma=(0.1,0.2),p=0.5):
        self.gaussian = GaussianBlur(kernel,sigma)
        self.permute = Permute([1,0,2,3])
        self.probability = p
        self.t = torchvision.transforms.Compose([self.permute, self.gaussian, self.permute])

    def __call__(self, clip):
        if self.probability < torch.rand(1):
            return clip
        else:
            blurred_clip = self.t(clip)
            return blurred_clip
    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.probability})"

class RandomRot90Video(object):
    '''
    Rotate a video by 90 degrees k-times in the image plane. Assume video is of a form
    such that the spatial plane is the last two dims [..., H, W]
    '''
    def __init__(self, k=None, p=0.5):
        self.probability = p
        self.k = k

    def __call__(self, clip):
        if self.probability < torch.rand(1):
            return clip
        else:
            if not self.k:
                self.k = torch.randint(1,4,(1,)).item()
            rot_clip = torch.rot90(clip, dims=[-2,-1],k=self.k)
            return rot_clip
    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.probability})"

class RandomVerticalFlipVideo(object):
    """
    Flip the video clip along the vertical direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if self.probability < torch.rand(1):
            return clip
        else:
            flip_clip = clip.flip(-2) # see vflip func in the torchvision functional_tensor docs https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
            return flip_clip

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.probability})"

class VarianceImageTransform(torch.nn.Module):
    def __init__(self, var_dim=1):
        super().__init__()
        assert var_dim in [1, 2]
        self.var_dim = var_dim
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def forward(self, clip):
        gray = torch.squeeze(clip[[0], ...])
        var = gray.var(axis=0).numpy()
        opening = cv2.morphologyEx(var, cv2.MORPH_OPEN, self.kernel)
        erode = cv2.erode(opening, self.ekernel, iterations=2)
        dilate_var = torch.tensor(cv2.dilate(erode, self.kernel, iterations=10))
        if self.var_dim == 2:
            var_array = torch.stack(
                    (gray, torch.stack([dilate_var] * gray.shape[0]), torch.stack([dilate_var] * gray.shape[0])))
        else:
            var_array = torch.stack((gray, gray, torch.stack([dilate_var] * gray.shape[0])))
        return var_array

