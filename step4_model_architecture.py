import torch
import torch.nn as nn

# --- PART 1: The Vision Part (U-Net Encoder) ---
# This looks at the Satellite Image to find forests, roads, and terrain.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- PART 2: The Time Part (LSTM) ---
# This looks at the past 24 hours of Wind and Temp.
class WeatherLSTM(nn.Module):
    def __init__(self, input_features=4, hidden_size=64):
        super().__init__()
        # input_features = 4 (Temp, Humidity, Wind Speed, Wind Direction)
        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x shape: (Batch, 24 hours, 4 features)
        # We only care about the final hidden state (the context after 24 hours)
        _, (h_n, _) = self.lstm(x)
        context_vector = h_n[-1] # Shape: (Batch, hidden_size)
        return self.fc(context_vector)

# --- PART 3: The Fusion (Putting it together) ---
class GeoTemporalFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Image Encoder (Downsampling)
        self.inc = DoubleConv(3, 64)           # Input: 3 channels (RGB)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # 2. Weather Encoder
        self.weather_net = WeatherLSTM(input_features=4, hidden_size=128)
        
        # 3. Fusion Bottleneck
        # We combine 512 image features + 128 weather features
        self.bottleneck_conv = DoubleConv(512 + 128, 512)
        
        # 4. Decoder (Upsampling to create the Map)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256) # 512 because we add skip connection
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, 1, kernel_size=1) # Output: 1 Channel (Fire Risk)

    def forward(self, image, weather):
        # A. Encode Image
        x1 = self.inc(image)       # 256x256
        x2 = self.down1(x1)        # 128x128
        x3 = self.down2(x2)        # 64x64
        x4 = self.down3(x3)        # 32x32 (Bottleneck Spatial)
        
        # B. Encode Weather
        w_vec = self.weather_net(weather) # Shape: (Batch, 128)
        
        # C. Fuse
        # We need to stretch the weather vector to fit the 32x32 image feature map
        b, c, h, w = x4.shape
        # Reshape weather to (Batch, 128, 1, 1) and expand to (Batch, 128, 32, 32)
        w_expanded = w_vec.view(b, -1, 1, 1).expand(b, -1, h, w)
        
        # Stick them together
        fused = torch.cat([x4, w_expanded], dim=1) 
        x4_new = self.bottleneck_conv(fused)
        
        # D. Decode (Reconstruct the fire map)
        x = self.up1(x4_new)
        x = torch.cat([x, x3], dim=1) # Skip connection
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # --- SYSTEM CHECK ---
    print("Testing Model Architecture...")
    
    # 1. Create a fake image (1 sample, 3 channels, 256x256 pixels)
    dummy_image = torch.randn(1, 3, 256, 256)
    
    # 2. Create fake weather history (1 sample, 24 hours, 4 features)
    dummy_weather = torch.randn(1, 24, 4)
    
    # 3. Initialize Model
    model = GeoTemporalFusionNet()
    print(" [✓] Model created successfully.")
    
    # 4. Run data through the brain
    output = model(dummy_image, dummy_weather)
    
    print(f" [✓] Input Image Shape: {dummy_image.shape}")
    print(f" [✓] Input Weather Shape: {dummy_weather.shape}")
    print(f" [✓] Output Prediction Shape: {output.shape}")
    
    if output.shape == (1, 1, 256, 256):
        print("\nSUCCESS! The architecture is ready for training.")
    else:
        print("\nERROR: Output shape is wrong.")