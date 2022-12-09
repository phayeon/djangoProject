import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
import sys
import dlib  # conda install -c conda-forge dlib
import cv2
import openface
'''
--- openface install ---
mkdir openface
cd openface
git clone https://github.com/cmusatyalab/openface.git ~/openface
cd ./~
cd openface
python setup.py install
'''

'''
https://arxiv.org/abs/1511.06434
Unsupervised Representation Learning 
with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz, Soumith Chintala
In recent years, supervised learning with convolutional networks (CNNs) 
has seen huge adoption in computer vision applications. 
Comparatively, unsupervised learning with CNNs has received less attention. 
In this work we hope to help bridge the gap 
between the success of CNNs for supervised learning and unsupervised learning. 
We introduce a class of CNNs called 
deep convolutional generative adversarial networks (DCGANs), 
that have certain architectural constraints, and demonstrate 
that they are a strong candidate for unsupervised learning. 
Training on various image datasets, we show convincing evidence 
that our deep convolutional adversarial pair learns a hierarchy of representations 
from object parts to scenes in both the generator and discriminator. 
Additionally, we use the learned features for novel tasks 
- demonstrating their applicability as general image representations.
'''

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DcGan(object):
    def __init__(self):
        # Root directory for dataset
        self.dataroot = r"C:\Users\AIA\PycharmProjects\djangoProject\movie\movies\data"
        # Number of workers for dataloader
        self.workers = 2
        # Batch size during training
        self.batch_size = 128
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = 64
        # Number of channels in the training images. For color images this is 3
        self.nc = 3
        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100
        # Size of feature maps in generator
        self.ngf = 64
        # Size of feature maps in discriminator
        self.ndf = 64
        # Number of training epochs
        self.num_epochs = 10
        # Learning rate for optimizers
        self.lr = 0.0002
        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = 1

    def celeb_a(self):
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = dset.ImageFolder(root=self.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Set random seed for reproducibility
        manualSeed = 999
        print("Random Seed: ", manualSeed)

        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=self.workers)
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

    def hook(self):
        self.show_face()
        self.print_netG()
        self.print_netD()
        self.generate_fake_faces()

    def show_face(self):
        # Plot some training images
        self.celeb_a()
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def print_netG(self):
        # Create the generator
        self.netG = Generator(self.ngpu).to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        self.netG.apply(self.weights_init)

        # Print the model
        print(self.netG)

    def print_netD(self):
        # Create the Discriminator
        self.netD = Discriminator(self.ngpu).to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(self.weights_init)

        # Print the model
        print(self.netD)

    def face_learn(self):
        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # Training Loop

        # Lists to keep track of progress
        self.img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(tqdm(self.dataloader)):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
            # Output training stats
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, self.num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    def generate_fake_faces(self):
        self.celeb_a()
        self.print_netG()
        self.print_netD()
        self.face_learn()
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.dataloader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        plt.show()



class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.that = DcGan()
        nz = self.that.nz
        ngf = self.that.ngf
        nc = self.that.nc
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.that = DcGan()
        ndf = self.that.ndf
        nc = self.that.nc

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class MyDlib(object):

    def __init__(self):
        pass

    def hook(self):
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        # 학습된 랜드마크 모델 데이터 경로
        predictor_model = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\data\shape_predictor_68_face_landmarks.dat"

        # HOG 이용한 얼굴 감지 클래스 생성 - dlib
        face_detector = dlib.get_frontal_face_detector()

        # 얼굴에 랜드마크 찾는 클래스 생성 - dlib
        # 매개변수로 랜드마크 모델
        face_pose_predictor = dlib.shape_predictor(predictor_model)

        # 랜드마크를 이용해 얼굴을 정렬할 클래스 생성 - Openface
        # 매개변수로 랜드마크 모델
        face_aligner = openface.AlignDlib(predictor_model)

        # 첫번째 매개변수로 부터 전달 받은 이미지 파일 경로
        file_name = r"C:\Users\AIA\PycharmProjects\djangoProject\movie\movies\data\celeba\000001.jpg"

        # 이미지 파일 경로로 부터 이미지(numpy.ndarry) 불러오기
        image = cv2.imread(file_name)

        '''
         이미지에서 얼굴 찾기
         얼굴 인식 두번째 변수 1은 업샘플링을 한번 하겠다는 얘기인데
         업샘플링을하면 더 많이 인식할 수 있다고 한다.
         다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
         그냥 1이면 될 듯. 
        '''

        detected_faces = face_detector(image, 1)

        '''
         detected_faces는 얼굴을 찾은 위치를 가지고 있는 
         "list of rectagles"로 rect형 리스트라고 보면 된다.
         이미지 파일에서 찾은 얼굴 개수 표시 
        '''
        print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

        # 찾은 얼굴 개수 만큼 반복한다.
        for i, face_rect in enumerate(detected_faces):
            '''
            찾은 얼굴 인댁스, 왼쪽, 위, 오른쪽, 아래 위치 (사각형)표시 
            '''
            print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i,
                                                                            face_rect.left(), face_rect.top(),
                                                                            face_rect.right(),
                                                                            face_rect.bottom()))

            # 얼굴 위치에서 랜드마크 찾기
            pose_landmarks = face_pose_predictor(image, face_rect)
            alignedFace = face_aligner.align(532, image, face_rect,
                                             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            '''
            alignedFace 는 RGB(ndarray) 이미지      
            '''

            # aligned_face_x.jpg 로 저장
            cv2.imwrite(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\data\aligned_face_{}.jpg".format(i), alignedFace)


dc_menu = ["Exit", #0
                "/movie/movies/fake-Image",# 1. Loading CelebA Dataset
                "/movie/movies/face-blow-up",#2. Blow Up Face By DLib
                "",#3
                "",#4
                "",#5
                "",#6
                ]

dc_lambda = {
    "1": lambda x: x.hook(),
    "2": lambda x: x.hook()
}

if __name__ == '__main__':
    ls = []
    while True:
        [print(f'{i}.{j}') for i, j in enumerate(dc_menu)]
        menu = input('Choose Menu : ')
        if menu == '0':
            print('종료')
            break
        else:
            if menu == "1":
                d = DcGan()
            else:
                d = MyDlib()
            dc_lambda[menu](d)
