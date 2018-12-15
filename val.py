import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from cycle_gan import Generator
from scipy.misc import imsave
import argparse


def save_image(image_tensor, path, pre):
    un_normalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    image = un_normalize(image_tensor).cpu().detach().numpy()
    image = image.transpose([1, 2, 0]) * 255

    if pre is None:
        imsave(path, image)
    else:
        imsave(os.path.join(pre, path), image)


def val(image_path, model_path, out_path=None, image_size=256, type='photo2anime'):
    model = Generator(3, 3)
    if type == 'photo2anime':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_x_anime.pth')))
    elif type == 'anime2photo':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_y_anime.pth')))
    elif type == 'photo2monet':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_y_monet.pth')))
    elif type == 'monet2photo':
        model.load_state_dict(torch.load(os.path.join(model_path, 'G_x_monet.pth')))
    else:
        raise Exception('type must be photo2anime , anime2photo, photo2monet or monet2photo')

    if image_size != 128 and image_size != 256:
        raise Exception('image_size must be 128 or 256')

    transform = transforms.Compose(
        [transforms.Resize(image_size, interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(image_path)
    image = transform(image)

    image = torch.unsqueeze(image, 0)
    image = image.float()
    fake_image = model(image)
    if out_path is None:
        save_image(fake_image[0], 'fake_image.jpg', os.path.dirname(image_path))
    else:
        save_image(fake_image[0], out_path, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True)
    parser.add_argument('--model_path', '-m', required=False, default='./pretrained_model')
    parser.add_argument('--out_path', '-o', required=False)
    parser.add_argument('--type', '-t', help='style transfer type: photo2anime,anime2photo,photo2monet,monet2photo',
                        required=True)
    parser.add_argument('--size', '-s', help='image size: 128 or 256', required=False, type=int, default=256)
    args = parser.parse_args()

    val(args.image_path, args.model_path, out_path=args.out_path, type=args.type, image_size=args.size)
