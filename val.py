import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from cycle_gan import Generator
from scipy.misc import imsave


def save_image(image_tensor, path, pre):
    un_normalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    image = un_normalize(image_tensor).cpu().detach().numpy()
    image = image.transpose([1, 2, 0]) * 255
    imsave(os.path.join(pre, path), image)


def val(image_path, model_path, type='photo2anime'):
    model = Generator(3, 3)
    if type == 'photo2anime':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_x_anime.pth')))
    elif type == 'anime2photo':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_y_anime.pth')))
    else:
        raise Exception('type must be photo2anime or anime2photo')

    transform = transforms.Compose(
        [transforms.Resize(128, interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(image_path)
    image = transform(image)

    image = torch.unsqueeze(image, 0)
    image = image.float()
    fake_image = model(image)
    save_image(fake_image[0], 'fake_image.jpg', pre=os.path.dirname(image_path))


if __name__ == '__main__':
    val('/home/meidai/图片/obama.jpg', './pretrained_model')
