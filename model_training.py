import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from custom_dataset import RealFakeDataset
import math
import random
import argparse
from ssim_torch_geometry import SSim
import numpy as np
from skimage import io
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches








############### Parser to ease the training
parser = argparse.ArgumentParser(description='PyTorch Welding Recurrent GAN')

parser.add_argument('--z_dim', type=int, default=100,
                    help='random z vector dimension')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')

parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')

parser.add_argument('--learning_rate', type=float, default=9e-4,
                    help='learning rate')

parser.add_argument('--loss_function', type=str,  default='hinge',
                    help='loss function to train GAN, hinge or wasserstein')

parser.add_argument('--directory_fake', type=str,  default='fake',
                    help='directory to save the generated images')

parser.add_argument('--directory_real', type=str,  default='real',
                    help='directory to save the real images from the data')

parser.add_argument('--directory_result', type=str,  default='num_result',
                    help='directory to save the numerical results from the code such as SSIM, MSE and the training process')

parser.add_argument('--input_size', type=int, default=1180,
                    help='input size at the NC-GRU/GRU')

parser.add_argument('--sequence_length', type=int, default=8,
                    help='number of bottom images')

parser.add_argument('--hidden_size', type=int, default=256,
                    help='hidden dimension at NC-GRU/GRU')

parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers NC-GRU/GRU')

parser.add_argument('--wave_current_dimension', type=int, default=30,
                    help='dimension of the vector of wave current information')

parser.add_argument('--bottom_images', type=int, default=60,
                    help='number of consecutive bottom images')

parser.add_argument('--train_data_dir', type=str,  default='Your_data_directory_to_train',
                    help='directory with topside and bottom images and current')


args = parser.parse_args()



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  




# Hyperparameters
z_dim = args.z_dim                                             #100 # it is 100 but 64 is from the image from the bottom
batch_size = args.batch_size                                   #32
num_epochs = args.num_epochs                                   #200
learning_rate = args.learning_rate                             #9e-4
loss_function = args.loss_function                             #'hinge'
directory_fake = args.directory_fake                           #'fake'
directory_real = args.directory_real                           #'real'
directory_result = args.directory_result                       #'result'
input_size = args.input_size                                   #1180 # 1024 + 156
sequence_length = args.sequence_length                         #8
hidden_size = args.hidden_size                                 #256
num_layers = args.num_layers                                   #1 # GRU

wave_current_dimension = args.wave_current_dimension           # vector dimension 30
bottom_images = args.bottom_images                             # 60 bottom images
train_data_dir = args.train_data_dir




#### Creating the folders for saving the images and numerical results
if not os.path.exists(directory_result):
    os.makedirs(directory_result)
if not os.path.exists(directory_fake):
    os.makedirs(directory_fake)
if not os.path.exists(directory_real):
    os.makedirs(directory_real)




# Data proprecessing.
transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5),
        #    (0.5))
            ])
            

# Load dataset
dataset = RealFakeDataset(
            real_dir=f"/{train_data_dir}/real",
            fake_dir=f"/{train_data_dir}/noise",
            current_dir=f"/{train_data_dir}",
            num_images=bottom_images,
            curr_dim=wave_current_dimension,
            transform=transforms.ToTensor(),
            )

def print_to_file(file1, path):
	f = open(path, "a+")
	f.write('{:}\n'.format(file1))
	f.close()




# train and test data
train_set, test_set = torch.utils.data.random_split(dataset, 
                    [len(dataset)-int(math.floor(0.2*len(dataset))), int(math.floor(0.2*len(dataset)))])

# Create dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)



# Define the generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=(4,4),stride=(2,2), padding=1) # out from 64x64x1 is 32x32x4
        self.conv2=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=(4,4),stride=(2,2), padding=1) # out 16x16x8
        self.batch_norm_pre1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(4,4),stride=(2,2), padding=1) # out 8x8x16   1024
        self.batch_norm_pre2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.deconv1=nn.ConvTranspose2d(in_channels=1,out_channels=16,kernel_size=(4,4),stride=(2,2),padding=1)
        self.deconv2=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=(4,4),stride=(2,2),padding=1)
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=False
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        
    def forward(self, x, current):
        
        current_list = []
        
        for i in range(bottom_images):
            current_new = self.conv1d_1(current[:, i, :].unsqueeze(1))
            current_new = self.conv1d_2(current_new)
            current_new = current_new.reshape(current_new.size(0), 1, -1)
            current_list.append(current_new)
        current_out = torch.cat(current_list, dim=1)
        
      
        h0 = torch.randn(self.num_layers , x.size(0), self.hidden_size).to(device) # similar to the z vector
        a_list = []
        
        
        for i in range(bottom_images):
            a = self.conv1(x[:, i, :, :].unsqueeze(1))
            a = self.batch_norm_pre1(self.conv2(a))
            a = self.batch_norm_pre2(self.conv3(a))
            a = a.reshape(x.size(0), 1, -1)
            a_list.append(a)
        x = torch.cat(a_list, dim=1)
        x = torch.cat((x, current_out), dim=2)
        

        out, _ = self.gru(x, h0)
        out_for_cnn=out[:,-1,:].reshape(-1,1,16,16)
        out_for_cnn = self.batch_norm1(self.deconv1(out_for_cnn))
        out_for_cnn = torch.tanh(self.deconv2(out_for_cnn))
        # out = self.fc(out[:, -1, :])

        return out_for_cnn
        
# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=bottom_images + 1,out_channels=16,kernel_size=(4,4),stride=(2,2), padding=1) # at the in_channels 1 is from image from top and 8 from image from bottom
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=(2,2), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=(2,2), padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=(2,2), padding=1)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=(4,4),stride=(2,2), padding=0)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batch_norm1(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm2(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm3(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        return x


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            

# Create generator and discriminator
generator = Generator(input_size, hidden_size, num_layers).to(device)
discriminator = Discriminator().to(device)


# Initialize the weights of generator and discriminator
initialize_weights(generator)
initialize_weights(discriminator)


# Use binary cross entropy loss and MSE loss for numerical results
loss_fn = nn.BCELoss()
mse_saved = nn.MSELoss()

# Use Adam optimizer for both the generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.7, verbose=True)                   #### Scheduler for Generator
scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.7, verbose=True)               #### Scheduler for Discriminator


# Training loop
for epoch in range(num_epochs):
    
    for i, (images, noise, current) in enumerate(train_loader):
        
        images = images.type(torch.float)
        noise = noise.type(torch.float)
        current = current.type(torch.float)
        images = images.to(device)
        noise = noise.to(device)
        current = current.to(device)
        
        # Generate fake images
        fake_images = generator(noise, current)
        disc_inp_fake = torch.cat([fake_images, noise.squeeze()], dim=1)
        
        
        if epoch % 40 == 0 and epoch != 0:
            if i == 0:
                scheduler_gen.step()
                scheduler_dis.step()
        
        if i == len(train_loader) - 1:
            print_to_file(f'Probability image {i} epoch {epoch}: {discriminator(disc_inp_fake).squeeze().detach()}', '{}/disc_output.txt'.format(directory_result))

        
        # Calculate loss for real images
        disc_inp = torch.cat([images, noise.squeeze()], dim=1)
        if loss_function == 'hinge':
            discriminator_loss = nn.ReLU()(1.0 - discriminator(disc_inp)).mean() + nn.ReLU()(1.0 + discriminator(disc_inp_fake.detach())).mean()
        elif loss_function == 'wasserstein':
            discriminator_loss = -discriminator(disc_inp).mean() + discriminator(disc_inp_fake.detach()).mean()
        
        # Train the discriminator
        discriminator_optimizer.zero_grad()
        
        # Backpropagate and update weights
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Train the generator
        generator_optimizer.zero_grad()
        
        # Calculate loss for fake images
        if loss_function == 'hinge' or loss_function == 'wasserstein':
            generator_loss_one = -discriminator(disc_inp_fake).mean()
        
        ssim_plus = SSim(fake_images, images, window_size=11)
        generator_loss = 0.2 * generator_loss_one + 0.8 * ssim_plus             #### Generator weighted loss 
        
        
        # Backpropagate and update weights
        generator_loss.backward()
        generator_optimizer.step()
        
        # Print progress
        if (i+1) % 100 == 0:
            message = f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}'
            print(message)
        
            print_to_file(message, '{}/output_file.txt'.format(directory_result))
        
        
        
        
    with torch.no_grad():
        test_loss_test = 0

        # Testing loop inside the training loop
        for test_images, test_noise, test_current in test_loader:
            
            test_images = test_images.type(torch.float)
            test_noise = test_noise.type(torch.float)
            test_current = test_current.type(torch.float)
            
            # Move images to device
            test_images = test_images.to(device)
            test_noise = test_noise.to(device)
            test_current = test_current.to(device)
            
    
            
    
            # Generate fake images
            test_fake_images = generator(test_noise, test_current)
            ssim_score = SSim(test_fake_images, test_images, window_size=11)
            print_to_file(f'Test Generator SSIM: {ssim_score:.4f} epoch {epoch}', '{}/ssim.txt'.format(directory_result))
            
            mse_score = mse_saved(test_images, test_fake_images)
            print_to_file(f'Test Generator MSE: {mse_score:.4f} epoch {epoch}', '{}/mse.txt'.format(directory_result))
    
            # Calculate loss for fake images
            test_images_loss = torch.cat([test_fake_images, test_noise.squeeze()], dim=1)
            test_generator_loss = loss_fn(discriminator(test_images_loss), torch.ones_like(discriminator(test_images_loss)).to(device))
            
            test_loss_test += test_generator_loss
    
        #### Print test loss
        print(f'Test Generator Loss: {test_loss_test.item()/len(test_loader):.4f}')
        print_to_file(f'Test Generator Loss: {test_loss_test.item()/len(test_loader):.4f}', '{}/output_file.txt'.format(directory_result))
        
        #### Save 15 generated images
        image_test = test_fake_images[:15].reshape(5*64, 3*64)
        real_image_test = test_images[:15].reshape(5*64, 3*64)
        
        
        
        vutils.save_image(image_test.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(directory_fake, epoch), normalize=True)
        if epoch == 0:
            vutils.save_image(real_image_test.detach(), '{}/real_samples.png'.format(directory_real), normalize=True)
        
        
        
