import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(outChannels)
        self.conv2 = nn.Conv3d(outChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class DeConvLayer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DeConvLayer, self).__init__()
        self.deconv = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn(self.deconv(x)))
        return out


class ConvAutoEncoder(nn.Module):
    def __init__(self, nChannels):
        super(ConvAutoEncoder, self).__init__()
        # encoder
        self.Conv1 = ConvBlock(1, nChannels)
        self.Conv2 = ConvBlock(nChannels, nChannels*2)
        self.Conv3 = ConvBlock(nChannels*2, nChannels*4)
        self.Conv4 = ConvBlock(nChannels*4, nChannels*8)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.pad = nn.ReplicationPad3d((3, 3, 1, 1, 3, 3))  # (left, right, top, bottom, front, back)

        # decoder
        self.deConv1 = DeConvLayer(nChannels*8, nChannels*4)
        self.Conv5 = ConvBlock(nChannels*4, nChannels*4)
        self.deConv2 = DeConvLayer(nChannels*4, nChannels*2)
        self.Conv6 = ConvBlock(nChannels*2, nChannels*2)
        self.deConv3 = DeConvLayer(nChannels*2, nChannels)
        self.Conv7 = ConvBlock(nChannels, nChannels)
        self.Conv8 = nn.Conv3d(nChannels, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.AvgPool(self.Conv1(x))
        feature = self.AvgPool(self.Conv2(feature))
        feature = self.AvgPool(self.Conv3(feature))
        feature = self.deConv1(self.Conv4(feature))
        feature = self.deConv2(self.Conv5(feature))
        feature = self.deConv3(self.Conv6(feature))
        out = self.Conv8(self.Conv7(feature))
        #out = self.sigmoid(out)
        return out


class Feature_Extraction(nn.Module):
    def __init__(self, nChannels):
        super(Feature_Extraction, self).__init__()
        # encoder for feature extraction
        self.Conv1 = ConvBlock(1, nChannels)
        self.Conv2 = ConvBlock(nChannels, nChannels*2)
        self.Conv3 = ConvBlock(nChannels*2, nChannels*4)
        self.Conv4 = ConvBlock(nChannels*4, nChannels*8)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feature = self.AvgPool(self.Conv1(x))
        feature = self.AvgPool(self.Conv2(feature))
        out_m = self.Conv3(feature)
        feature = self.AvgPool(out_m)
        out_s = self.Conv4(feature)
        return out_s, out_m


'''
# ConvAutoEncoder 모델 학습 후 파라미터 저장
autoencoder = ConvAutoEncoder(nChannels)
# autoencoder.load_state_dict(...) # 학습된 파라미터 로드

# Feature_Extraction 모델 생성
feature_extractor = Feature_Extraction(nChannels)

# ConvAutoEncoder의 인코더 부분 파라미터를 Feature_Extraction으로 복사
feature_extractor.Conv1.load_state_dict(autoencoder.Conv1.state_dict())
feature_extractor.Conv2.load_state_dict(autoencoder.Conv2.state_dict())
feature_extractor.Conv3.load_state_dict(autoencoder.Conv3.state_dict())
feature_extractor.Conv4.load_state_dict(autoencoder.Conv4.state_dict())
feature_extractor.AvgPool = autoencoder.AvgPool  # AvgPool 레이어는 상태가 없으므로 직접 할당

# 이제 feature_extractor를 사용하여 특성 추출 가능
with torch.no_grad():
    out_s, out_m = feature_extractor(input_data)

'''