import torch
import torch.nn as nn

# --- Bloc d'Attention (Squeeze-and-Excitation) ---
# Ce bloc permet au modèle d'apprendre quels canaux (par exemple, quelles couleurs ou quelles textures) 
# sont les plus importants pour restaurer l'image. Il "compresse" l'information (Squeeze) pour avoir 
# une vue globale, puis "excite" (Excitation) les canaux essentiels en leur donnant plus de poids.
# C'est particulièrement utile sous l'eau où le canal rouge nécessite souvent plus d'attention.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # La phase descendante (Encodeur) extrait les informations de l'image.
        # On sauvegarde les résultats (skip1, skip2, skip3) avant chaque réduction de taille (pool).
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Bloc de Double Convolution ---
# Brique fondamentale du U-Net. Chaque étape de l'encodeur et du décodeur 
# passe par deux couches de convolution successives pour extraire les caractéristiques.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Intégration de l'attention spatiale après chaque double convolution
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        # La phase descendante (Encodeur) extrait les informations de l'image.
        # On sauvegarde les résultats (skip1, skip2, skip3) avant chaque réduction de taille (pool).
        x = self.double_conv(x)
        x = self.se_block(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Unet, self).__init__()
        
        # --- ENCODEUR ---
        self.pool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # --- DÉCODEUR ---
        # Pour agrandir l'image dans le décodeur, on utilise une interpolation bilinéaire (Upsample) 
        # suivie d'une convolution standard. Cette méthode est préférée à la "ConvTranspose2d" classique 
        # car elle permet d'éviter l'effet damier (checkerboard artifacts) très fréquent lors de la génération d'images.
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1)
        )
        self.conv_up4 = DoubleConv(512, 256)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1)
        )
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up1 = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # La phase descendante (Encodeur) extrait les informations de l'image.
        # On sauvegarde les résultats (skip1, skip2, skip3) avant chaque réduction de taille (pool).
        skip1 = self.down1(x)
        x = self.pool(skip1)
        
        skip2 = self.down2(x)
        x = self.pool(skip2)
        
        skip3 = self.down3(x)
        x = self.pool(skip3)
        
        bottleneck = self.down4(x)
        
        # La phase montante (Décodeur) reconstruit l'image.
        # À chaque étape, on concatène (torch.cat) l'image agrandie avec la sauvegarde correspondante
        # de l'encodeur (les "skip connections"). C'est ce qui permet au U-Net de ne pas perdre
        # les détails spatiaux (les contours fins) pendant la reconstruction.
        x = self.up4(bottleneck)
        x = torch.cat((x, skip3), dim=1)
        x = self.conv_up4(x)
        
        x = self.up3(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.conv_up3(x)
        
        x = self.up2(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.conv_up2(x)
        
        x = self.up1(x)
        return x