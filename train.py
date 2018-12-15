import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from cycle_gan import Generator, Discriminator
from scipy.misc import imsave
import argparse


class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class ImageDataset(data.Dataset):
    def __init__(self, x_path, y_path, transform):
        self.transform = transform

        self.x_names = ImageDataset.find_image(x_path)
        self.y_names = ImageDataset.find_image(y_path)
        random.shuffle(self.x_names)
        random.shuffle(self.y_names)

        self.size = min(len(self.x_names), len(self.y_names))
        self.x_images = {}
        self.y_images = {}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x_name = self.x_names[index]
        y_name = self.y_names[index]

        if x_name in self.x_images:
            x = self.x_images[x_name]
        else:
            x = Image.open(x_name)
            self.x_images[x_name] = x

        if y_name in self.y_images:
            y = self.y_images[y_name]
        else:
            y = Image.open(y_name)
            self.y_images[y_name] = y

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    @staticmethod
    def find_image(path):
        names = os.listdir(path)
        image_names = []
        for name in names:
            name_path = os.path.join(path, name)
            if os.path.isdir(name_path):
                sub_names = ImageDataset.find_image(name_path)
                image_names += sub_names
            elif name_path.endswith('jpg') or name_path.endswith('png') or name_path.endswith('jpeg'):
                image_names.append(name_path)

        return image_names


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch - 99) / 101
    return lr_l


def save_image(image_tensor, path, pre):
    un_normalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    image = un_normalize(image_tensor).cpu().detach().numpy()
    image = image.transpose([1, 2, 0]) * 255
    imsave(os.path.join(pre, path), image)


def train(x_path, y_path, image_size=128, log_path='./log', model_path='./model'):
    batch_size = 1
    learning_rate = 0.0002
    max_epoch = 200
    pool_size = 50

    transform_train = transforms.Compose(
        [transforms.Resize(int(image_size * 1.1), interpolation=Image.BICUBIC),
         transforms.RandomCrop(image_size),
         transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(0.01, 0.01, 0.01, 0.01),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageDataset(x_path, y_path, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    print('train:{}'.format(len(dataset)))

    G_x = Generator(3, 3)
    G_y = Generator(3, 3)
    D_x = Discriminator(3)
    D_y = Discriminator(3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G_x = G_x.to(device)
    G_y = G_y.to(device)
    D_x = D_x.to(device)
    D_y = D_y.to(device)

    optimizer_G = optim.Adam([*G_x.parameters(), *G_y.parameters()], lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam([*D_x.parameters(), *D_y.parameters()], lr=learning_rate, betas=(0.5, 0.999))
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    criterionGAN = nn.MSELoss()
    criterionCycle = nn.L1Loss()

    fake_x_pool = ImagePool(pool_size)
    fake_y_pool = ImagePool(pool_size)

    for epoch in range(max_epoch):
        for i, data in enumerate(dataloader, 0):
            x, y = data
            if x.shape[1] != 3 or y.shape[1] != 3:
                continue
            x, y = x.float().to(device), y.float().to(device)

            fake_y = G_x(x)
            rec_x = G_y(fake_y)

            fake_x = G_y(y)
            rec_y = G_x(fake_x)

            #################### Generator ######################
            # G_x and G_y
            D_x.set_requires_grad(False)
            D_y.set_requires_grad(False)
            optimizer_G.zero_grad()
            # GAN loss
            pred_y = D_y(fake_y)
            pred_x = D_x(fake_x)
            real_label = torch.tensor(1.0).expand_as(pred_x).to(device)
            fake_label = torch.tensor(0.0).expand_as(pred_x).to(device)
            loss_G_x = criterionGAN(pred_y, real_label)
            loss_G_y = criterionGAN(pred_x, real_label)
            # cycle loss
            loss_cycle_x = criterionCycle(rec_x, x) * 10
            loss_cycle_y = criterionCycle(rec_y, y) * 10
            # G loss
            loss_G = loss_G_x + loss_G_y + loss_cycle_x + loss_cycle_y
            loss_G.backward()
            optimizer_G.step()

            #################### Discriminator ######################
            # D_x and D_y
            D_x.set_requires_grad(True)
            D_y.set_requires_grad(True)
            optimizer_D.zero_grad()
            # query history
            h_fake_x = fake_x_pool.query(fake_x)
            h_fake_y = fake_y_pool.query(fake_y)
            # loss D
            loss_D_x = (criterionGAN(D_x(x), real_label) + criterionGAN(D_x(h_fake_x.detach()), fake_label)) * 0.5
            loss_D_y = (criterionGAN(D_y(y), real_label) + criterionGAN(D_y(h_fake_y.detach()), fake_label)) * 0.5
            loss_D_x.backward()
            loss_D_y.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    'epoch:{},current:{}, loss_G_x:{:.2f},  loss_G_y:{:.2f},   loss_cycle_x:{:.2f},   loss_cycle_y:{:.2f},    loss_D_x:{:.2f},    loss_D_y:{:.2f}'
                        .format(epoch, i, loss_G_x.item(), loss_G_y.item(), loss_cycle_x.item(), loss_cycle_y.item(),
                                loss_D_x.item(), loss_D_y.item()))

            if i + 1 == len(dataloader):
                save_image(x[0], 'real_x_{}.jpg'.format(epoch), log_path)
                save_image(y[0], 'real_y_{}.jpg'.format(epoch), log_path)
                save_image(fake_y[0], 'fake_y_{}.jpg'.format(epoch), log_path)
                save_image(rec_x[0], 'rec_x_{}.jpg'.format(epoch), log_path)
                save_image(fake_x[0], 'fake_x_{}.jpg'.format(epoch), log_path)
                save_image(rec_y[0], 'rec_y_{}.jpg'.format(epoch), log_path)

        scheduler_G.step()
        scheduler_D.step()

        if epoch % 20 == 0:
            torch.save(G_x.state_dict(), os.path.join(model_path, 'G_x_{}.pth'.format(epoch)))
            torch.save(G_y.state_dict(), os.path.join(model_path, 'G_y_{}.pth'.format(epoch)))
            torch.save(D_x.state_dict(), os.path.join(model_path, 'D_x_{}.pth'.format(epoch)))
            torch.save(D_y.state_dict(), os.path.join(model_path, 'D_y_{}.pth'.format(epoch)))

    torch.save(G_x.state_dict(), os.path.join(model_path, 'G_x.pth'))
    torch.save(G_y.state_dict(), os.path.join(model_path, 'G_y.pth'))
    torch.save(D_x.state_dict(), os.path.join(model_path, 'D_x.pth'))
    torch.save(D_y.state_dict(), os.path.join(model_path, 'D_y.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xpath', '-x', help='Path to training set, x style of images', required=True)
    parser.add_argument('--ypath', '-y', help='Path to training set, y style of images', required=True)
    parser.add_argument('--size', '-s', help='image size', required=False, type=int, default=256)
    args = parser.parse_args()

    train(args.xpath, args.ypath, image_size=args.size)
