# CSE499
# Captsone Project on Deep Convolutional Generative Adverserial Network for Bangla Handwritten Text Generation 


Earlier a lot of work has been done regarding Optical Character Recognition (OCR). Handwritten Text Generation (HTG) is a next level work in terms of machine learning technology. Though a few works have been done regarding handwritten text generation and recognition, generating the  handwritten texts in Bangla language has not been done yet. A few works have been done regarding Bangla handwritten text recognition but those were not that satisfying. We want to highlight the Bangla Language more to the people. So we found this interesting to work with. In this report we are proposing a deep Convolutional Neural Network  (CNN) and Generative Adversarial Network (GAN) to generate Bangla Handwritten Digits. In the beginning our motivation was not only generating Bangla numerals but also all Bangla basic and compound characters. 

So continuing like this we present our Bangla Handwritten digits generated using an unsupervised algorithm DCGAN which is a popular and successful network design for GAN. We worked with the architecture and tried changing them to see how various the results can be. 

We showed how to generate handwritten Bangla digits using DCGAN from Bangla plain text. 

## Discrimnator Architecture
![logo](https://github.com/tonygms2/CSE499/blob/main/GEN.png)

## Generator Architecture 
![logo](https://github.com/tonygms2/CSE499/blob/main/GEN.png)

## Real Image VS Genereated Image 
![logo](https://github.com/tonygms2/CSE499/blob/main/compare.PNG)

## Discrimnator Loss
![logo](https://github.com/tonygms2/CSE499/blob/main/DISLOSS.png)

## Generator Loss
![logo](https://github.com/tonygms2/CSE499/blob/main/GENLOSS.png)

# Generator Class
```python 

#Generator
class Generator(nn.Module):
    
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace = True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    
        return torch.randn(n_samples, z_dim, device=device)

```
# Discrimnator Class
```python

#  Discriminator
class Discriminator(nn.Module):
   
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.01,inplace=True)

            )
        else: 
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.LeakyReLU(0.01)

            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

```
