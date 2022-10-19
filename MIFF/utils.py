
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import torch


'''
返回图片的信息熵
输入图片格式为torch.tensor
'''
def _entropy_tensor(image_src):
    image_src = image_src * torch.log2(image_src)
    sum = torch.sum(image_src)
    return sum * -1.0


'''
返回图片的信息熵
输入图片格式为np.array
'''
def _entropy(image_src):
    # img_gray = rgb2gray(image_src)
    img_gray = image_src.convert('L')
    entropy = shannon_entropy(img_gray)
    return entropy


'''
返回两张图片的相似度
输入图片格式为np.array
'''
def similarity(image_a, image_b):
    multichannel = True if len(image_a.shape) == 3 and image_a.shape[2] == 3 else False
    return ssim(image_a, image_b, multichannel=multichannel)


'''
返回两张图片的相似度
输入图片格式为torch.tensor
'''
def similarity_tensor(image_a, image_b):
    epsilon = 1e-9
    up = 2 * torch.sum(image_a * image_b) + epsilon
    down = torch.sum(image_a * image_a + image_b * image_b) + epsilon


if __name__=="__main__":
    pass