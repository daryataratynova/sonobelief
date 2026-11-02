"""Small 2D convolutional autoencoder for plane observations.

M_P (C x 128 x 128) --E_img--> e_P (latent_dim) --D_img--> logits (C x 128 x 128)
"""
from torch import nn


def block(cin, cout, down=True):
    conv = (nn.Conv2d(cin, cout, 4, 2, 1) if down else nn.ConvTranspose2d(cin, cout, 4, 2, 1))
    return nn.Sequential(conv, nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, in_ch=2, latent_dim=128, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        chans = [in_ch, *widths]
        self.convs = nn.Sequential(*[block(a, b) for a, b in zip(chans[:-1], chans[1:])])  # 128 -> 4
        self.fc = nn.Linear(widths[-1] * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.convs(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, out_ch=2, latent_dim=128, widths=(256, 256, 128, 64, 32)):
        super().__init__()
        self.w0 = widths[0]
        self.fc = nn.Linear(latent_dim, widths[0] * 4 * 4)
        self.convs = nn.Sequential(*[block(a, b, down=False) for a, b in zip(widths[:-1], widths[1:])])  # 4 -> 64
        self.head = nn.Sequential(nn.ConvTranspose2d(widths[-1], 32, 4, 2, 1), nn.SiLU(),  # 64 -> 128
                                  nn.Conv2d(32, out_ch, 3, 1, 1))

    def forward(self, z):
        return self.head(self.convs(self.fc(z).view(-1, self.w0, 4, 4)))


class PlaneAE(nn.Module):
    def __init__(self, channels=2, latent_dim=128):
        super().__init__()
        self.enc = Encoder(channels, latent_dim)
        self.dec = Decoder(channels, latent_dim)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z
