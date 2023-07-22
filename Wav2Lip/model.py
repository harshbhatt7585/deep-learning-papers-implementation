import torch
import torch.nn as nn


''' SyncNet [9] inputs a window V of Tv
consecutive face frames (lower half only) and a speech segment
S of size Ta × D, where Tv and Ta are the video and audio timesteps respectively '''



class SyncNetDiscriminator(nn.Module):
    def __init__(self):
        super(SyncNetDiscriminator, self).__init__()

    
class SpeechEncoder(nn.Module):

    '''The audio encoder is a standard CNN that
    takes a Mel-frequency cepstral coefficient (MFCC) heatmap of size
    MxTx1 and creates an audio embedding of size h.'''

    ''' 
        input: BxMxTx1
        output: Bxh (h=256)
    '''

    def __init__(self, inp_channel=1, kernel_size=3):
        super(SpeechEncoder, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(inp_channel, 64, kernel_size, padding=1),
            nn.Conv2d(64, 128, kernel_size, padding=1),
            nn.MaxPool2d(kernel_size, stride=2),
            nn.Conv2d(128, 256, kernel_size, padding=1),
            nn.Conv2d(256, 256, kernel_size, padding=1),
            nn.Conv2d(256, 512, kernel_size, padding=1),
            nn.MaxPool2d(kernel_size, stride=2),
        ])

        self.audio_embedding = nn.ModuleList([
            nn.Linear((512*2*8), 512),
            nn.Linear(512, 256)
        ])
        self.flatten = nn.Flatten()

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        
        x = self.flatten(x)
        for linear in self.audio_embedding:
            x = linear(x) 
        return x


'''
The encoder consists of a
series of residual blocks with intermediate down-sampling layers
and it embeds the given input image into a face embedding of size
h.
'''

class FaceEncoder(nn.Module):

    '''
        input: BxHxHx6
        output: Bxh
    '''

    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(FaceEncoder, self).__init__()
        self.resblocks = nn.ModuleList(
                [
                    FaceResidualBlock(64, 64, kernel_size, stride, padding),
                    FaceResidualBlock(64, 64, kernel_size, stride, padding),
                    FaceResidualBlock(96, 96, kernel_size, stride, padding),
                    FaceResidualBlock(96, 96, kernel_size, stride, padding),
                    FaceResidualBlock(256, 256, kernel_size, stride, padding),
                    FaceResidualBlock(512, 512, kernel_size, stride, padding),
                    FaceResidualBlock(512, 512, kernel_size, stride, padding),
            ])
        
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(3, 64, 7, stride=2),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(64, 96, 5, stride=2),
                nn.Conv2d(96, 96, 3),
                nn.Conv2d(96, 256, 3),
                nn.Conv2d(256, 512, 3),
                nn.Conv2d(512, 512, 3),
            ]
        )

        self.face_emmedding = nn.ModuleList([
            nn.Linear((512*3*3), 512),
            nn.Linear(512, 256)
        ])
        self.flatten = nn.Flatten()
        self.skip_connections = []
        
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.resblocks[i](x)
            if i < 6:
                self.skip_connections += [x]

        x = self.flatten(x)
        for linear in self.face_emmedding:
            x = linear(x)

        return x 


class FaceResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size, stride=1, padding=1):
        super(FaceResidualBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channles, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(in_channels, out_channles)
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(1)

    def forward(self, x):
        o = self.conv(x)
        o = self.batch_norm(o)
        o = self.relu(o)
        o = self.conv(o)
        o = self.batch_norm(o)
        o = self.relu(o)
        return x + o
    
    

class FaceDecoder(nn.Module):
    def __init__(self, inp_dim):
        super(FaceDecoder, self).__init__()

        self.linear = nn.Linear(inp_dim, 128)

        self.transpose_convs = nn.ModuleList([
            nn.ConvTranspose2d(128, 512, 5, stride=2),
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.ConvTranspose2d(256, 96, 3, stride=1),
            nn.ConvTranspose2d(96, 96, 3, stride=1),
            nn.ConvTranspose2d(96, 64, 6, stride=2),
            nn.ConvTranspose2d(64, 3, 7, stride=2, padding=2),
        ])
    
    def forward(self, x, skip_connections):
        x = self.linear(x) 
        x = x.view(x.size(0), x.size(1), 1, 1)
        for idx, layer in enumerate(self.transpose_convs):
            x = layer(x)
            print(x.shape)
            if idx < 6:
                print(x.shape, skip_connections[5 - idx].shape)
                x += skip_connections[5 - idx]
        return x 
    


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        

fe = FaceEncoder()
rand1 = torch.randn((1, 3, 112, 112))
r1 = fe(rand1)
print(r1.shape)
se = SpeechEncoder()
rand2 = torch.randn((1, 1, 12, 35)) 
r2 = se(rand2)
print(r2.shape)
r = torch.cat((r1, r2), 1)
print(r.shape)
skip = fe.skip_connections
print(len(skip))
decode = FaceDecoder(512)
out = decode(r, skip)
print(out.shape)