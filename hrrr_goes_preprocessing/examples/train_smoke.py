"""One optimization step; validates tensors, not forecast skill.
Run: python examples/train_smoke.py demo_data
"""
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from forecastprep.torch_data import ForecastDataset, Standardizer


class Hybrid(nn.Module):
    def __init__(self, store):
        super().__init__()
        self.actions = store.actions
        self.encoder = nn.Sequential(nn.Conv2d(2,8,3,padding=1),nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1),nn.Flatten())
        size = 2*len(store.columns)+8
        for a in self.actions.values(): size += (2*len(a['columns']) if a['kind']=='tabular' else 8)+1
        self.mlp = nn.Sequential(nn.Linear(size,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        parts=[x['core'],x['core_mask'].float(),
               self.encoder(torch.cat([x['overview'],x['overview_mask'].float()],1))]
        for key,a in self.actions.items():
            obj=x['optional'][key]; mask=obj['acquired'].float().unsqueeze(1)
            if a['kind']=='tabular': feature=torch.cat([obj['value'],obj['valid'].float()],1)
            else: feature=self.encoder(torch.cat([obj['value'],obj['valid'].float()],1))
            parts.extend([feature*mask,mask])
        return self.mlp(torch.cat(parts,1)).squeeze(-1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--mode',choices=['direct','mos'],default='mos'); args=p.parse_args()
    torch.manual_seed(123)
    normalizer=Standardizer.fit(args.root)
    ds=ForecastDataset(args.root,args.mode,'train',normalizer,actions=['history','infrared_local'],budget=1.05)
    if len(ds)==0: raise ValueError('No labelled training examples')
    batch=next(iter(DataLoader(ds,batch_size=4,shuffle=True,num_workers=args.workers)))
    model=Hybrid(ds.store); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    optimizer.zero_grad(); prediction=model(batch['inputs'])
    loss=nn.functional.mse_loss(prediction,batch['target']); loss.backward(); optimizer.step()
    print({'batch_size':len(prediction),'mode':args.mode,'loss':float(loss.detach()),'finite':bool(torch.isfinite(loss))})

if __name__=='__main__': main()  # Required for Windows DataLoader workers.
