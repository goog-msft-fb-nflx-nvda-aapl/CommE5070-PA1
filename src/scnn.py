"""Short-Chunk CNN, following Won et al. "Evaluation of CNN-based automatic music
tagging models" (SMC 2020) / github.com/minzwon/sota-music-tagging-models.
Trained from scratch on short log-mel crops; multi-segment-averaged at inference.
"""
import torch
import torch.nn as nn
import torchaudio


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.mp(self.relu(self.bn(self.conv(x))))


class ShortChunkCNN(nn.Module):
    def __init__(self, n_class, sample_rate=24000, n_fft=512, n_mels=128, n_channels=128):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 2, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        c = n_channels
        self.layer1 = ConvBlock(1, c)
        self.layer2 = ConvBlock(c, c)
        self.layer3 = ConvBlock(c, c * 2)
        self.layer4 = ConvBlock(c * 2, c * 2)
        self.layer5 = ConvBlock(c * 2, c * 2)
        self.layer6 = ConvBlock(c * 2, c * 2)
        self.layer7 = ConvBlock(c * 2, c * 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense1 = nn.Linear(c * 4, c * 4)
        self.bn1 = nn.BatchNorm1d(c * 4)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(c * 4, n_class)
        self.relu = nn.ReLU()

    def embed(self, x):
        """x: (B, T) waveform -> (B, C) pooled feature before classifier head."""
        x = self.melspec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x):
        feat = self.embed(x)
        h = self.relu(self.bn1(self.dense1(feat)))
        h = self.dropout(h)
        return self.dense2(h)


def spec_augment(mel, freq_mask=16, time_mask_frac=0.1, n_freq_masks=2, n_time_masks=2):
    """Apply to a (B, 1, n_mels, T) log-mel batch in-place-ish (returns new tensor)."""
    mel = mel.clone()
    n_mels, T = mel.shape[-2], mel.shape[-1]
    time_mask = max(1, int(T * time_mask_frac))
    for _ in range(n_freq_masks):
        f0 = torch.randint(0, max(1, n_mels - freq_mask), (1,)).item()
        mel[..., f0:f0 + freq_mask, :] = 0
    for _ in range(n_time_masks):
        t0 = torch.randint(0, max(1, T - time_mask), (1,)).item()
        mel[..., t0:t0 + time_mask] = 0
    return mel


def mixup(waves, labels, n_class, alpha=0.2):
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(waves.size(0), device=waves.device)
    mixed = lam * waves + (1 - lam) * waves[idx]
    y_onehot = torch.zeros(waves.size(0), n_class, device=waves.device)
    y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed, y_mixed
