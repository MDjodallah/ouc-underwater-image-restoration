import torch
import torch.nn as nn

class DoubleConv (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # Bloc 1 : Conv -> Norm -> ReLU
            # in_channels : ce qui entre (ex: 3 au tout début)
            # out_channels : ce qu'on veut (ex: 64)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Bloc 2 : Conv -> Norm -> ReLU
            # On reste à out_channels car on a déjà fait le travail de montée en dimension
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Ici, tu appelles juste ton bloc
        return self.double_conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Unet, self).__init__()
        
        # --- ENCODEUR (Descente) ---
        self.pool = nn.MaxPool2d(2)
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512) # Le fond du U (Bottleneck)

        # --- DÉCODEUR (Montée) ---
        # Les escalators (ConvTranspose2d)
        self.transpose1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Les blocs de traitement après concaténation
        self.up4 = DoubleConv(512, 256) # 256 (escalator) + 256 (skip) = 512
        self.up3 = DoubleConv(256, 128) # 128 (escalator) + 128 (skip) = 256
        self.up2 = DoubleConv(128, 64)  # 64 (escalator)  + 64 (skip)  = 128
        
        # La couche finale (le pinceau RGB)
        self.up1 = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # --- ENCODEUR ---
        skip1 = self.down1(x)         # Sortie : 64 canaux (256x256) - On sauvegarde !
        x = self.pool(skip1)
        
        skip2 = self.down2(x)         # Sortie : 128 canaux (128x128) - On sauvegarde !
        x = self.pool(skip2)
        
        skip3 = self.down3(x)         # Sortie : 256 canaux (64x64) - On sauvegarde !
        x = self.pool(skip3)
        
        bottleneck = self.down4(x)    # Sortie : 512 canaux (32x32) - Le fond du U
        
        # --- DÉCODEUR ---
        # Remontée 1
        x = self.transpose1(bottleneck)        # Gonfle à 64x64 (256 canaux)
        x = torch.cat((x, skip3), dim=1)       # Collation ! -> 512 canaux
        x = self.up4(x)                        # Traitement -> 256 canaux
        
        # Remontée 2
        x = self.transpose2(x)                 # Gonfle à 128x128 (128 canaux)
        x = torch.cat((x, skip2), dim=1)       # Collation ! -> 256 canaux
        x = self.up3(x)                        # Traitement -> 128 canaux
        
        # Remontée 3
        x = self.transpose3(x)                 # Gonfle à 256x256 (64 canaux)
        x = torch.cat((x, skip1), dim=1)       # Collation ! -> 128 canaux
        x = self.up2(x)                        # Traitement -> 64 canaux
        
        # Image Finale
        x = self.up1(x)                        # 64 canaux -> 3 canaux (RGB)
        
        return x
    
if __name__ == "__main__":
    # Test d'ingénierie (Savoir si notre architecture compile sans bug de dimension)
    print("Création d'un faux batch de 2 images RGB (256x256)...")
    faux_batch = torch.randn(2, 3, 256, 256)
    
    # On instancie le modèle
    modele = Unet(in_channels=3, out_channels=3)
    
    try:
        # On fait passer l'image dans le réseau
        resultat = modele(faux_batch)
        print("✅ INCROYABLE ! Le U-Net fonctionne de A à Z !")
        print(f"Taille de sortie : {resultat.shape} (On veut : [2, 3, 256, 256])")
    except Exception as e:
        print(f"❌ Erreur de dimension : {e}")