"""Numeric MLP and masked satellite CNN/MLP predictors."""
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, width))

    def forward(self, values, valid):
        return self.layers(torch.cat([values, valid.float()], dim=1))


class ForecastModel(nn.Module):
    def __init__(self, core_size, actions, kind='hybrid', hidden=(128, 64), embedding=16, dropout=.1):
        super().__init__()
        if kind not in ('mlp', 'hybrid'):
            raise ValueError('Unknown model kind')
        self.actions, self.kind, self.embedding = actions, kind, embedding
        self.encoders = nn.ModuleDict()
        size = core_size * 2
        if kind == 'hybrid':
            self.encoders['overview'] = ImageEncoder(embedding)
            size += embedding + 1
        for name, action in actions.items():
            if action['kind'] == 'tabular':
                size += len(action['columns']) * 2 + 1
            elif kind == 'hybrid':
                self.encoders[name] = ImageEncoder(embedding)
                size += embedding + 2
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(size, width), nn.SiLU(), nn.Dropout(dropout)])
            size = width
        layers.append(nn.Linear(size, 1))
        self.head = nn.Sequential(*layers)

    def encode(self, name, values, valid, acquired):
        active = acquired.bool() & valid.flatten(1).any(dim=1)
        result = values.new_zeros((len(values), self.embedding))
        # Unacquired images never pass through the encoder.
        if active.any():
            result[active] = self.encoders[name](values[active], valid[active])
        return result

    def representation(self, inputs):
        pieces = [inputs['core'], inputs['core_mask'].float()]
        if self.kind == 'hybrid':
            values, valid = inputs['overview'], inputs['overview_mask']
            present = valid.flatten(1).any(dim=1)
            pieces.extend([self.encode('overview', values, valid, present), present.float()[:, None]])
        for name, action in self.actions.items():
            item = inputs['optional'][name]
            acquired = item['acquired'].float().reshape(-1, 1)
            if action['kind'] == 'tabular':
                pieces.extend([item['value'] * acquired, item['valid'].float() * acquired, acquired])
            elif self.kind == 'hybrid':
                pieces.extend([self.encode(name, item['value'], item['valid'], acquired[:, 0]),
                               item['valid'].float().flatten(1).mean(1, keepdim=True) * acquired, acquired])
        return torch.cat(pieces, dim=1)

    def forward(self, inputs):
        return self.head(self.representation(inputs)).squeeze(1)


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device) if isinstance(value, torch.Tensor) else value
