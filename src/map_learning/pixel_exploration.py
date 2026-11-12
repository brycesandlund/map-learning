"""Noisy composite pixels and pixel-only representation training.

Positions are renderer inputs, never learner inputs. Maze topology changes only
wall geometry; all room interiors remain reachable and coverage denominators valid.
"""
import copy
import numpy as np
import torch
from torch import nn
from .maze import Maze
from .models import mlp


class PixelMaze(Maze):
    def __init__(self, config, seed=0, family='maze', noise=.12, rgb=False, marker_size=.85, noise_mode="full"):
        self.family = family
        self.rgb=rgb;self.marker_size=marker_size
        self.noise_mode=noise_mode;self.viewport=config.image_size-(8 if noise_mode=="panel" else 0)
        self.noise = noise
        super().__init__(config, seed)
        r = config.image_size
        self.yy, self.xx = np.mgrid[:r, :r].astype(np.float32)

    def _generate_walls(self):
        if self.family == 'maze':
            return super()._generate_walls()
        n = self.config.size
        walls = [(0,0,n,0),(0,n,n,n),(0,0,0,n),(n,0,n,n)]
        if self.family == 'snake':
            # Alternating doors connect n long horizontal corridors in a chain.
            for y in range(1,n):
                walls.append((0,y,n-1,y) if y%2 else (1,y,n,y))
        elif self.family == 'comb':
            # Vertical dead ends join a single horizontal corridor at the bottom.
            walls += [(x,1,x,n) for x in range(1,n)]
        elif self.family == 'rooms':
            mid = n//2
            walls += [(mid,0,mid,1),(mid,2,mid,n),
                      (0,mid,1,mid),(2,mid,mid,mid),
                      (mid,mid,n-2,mid),(n-1,mid,n,mid)]
        elif self.family != 'open':
            raise ValueError(self.family)
        return np.array(walls,dtype=np.float32).reshape(-1,2,2)

    def _raster_background(self):
        if self.noise_mode!='panel':return super()._raster_background()
        canvas=np.full((self.config.image_size,self.config.image_size),.5,np.float32)
        canvas[:self.viewport,:self.viewport]=1
        for segment in self.walls:
            a,b=np.rint(segment/self.config.size*(self.viewport-1)).astype(int)
            canvas[min(a[1],b[1]):max(a[1],b[1])+1,min(a[0],b[0]):max(a[0],b[0])+1]=0
        return canvas

    @property
    def observation_shape(self):
        return (3 if self.rgb else 1,self.config.image_size,self.config.image_size)

    def observe(self, position=None):
        p = self.position if position is None else np.asarray(position)
        x,y = p / self.config.size * (self.viewport-1)
        marker = np.exp(-((self.xx-x)**2+(self.yy-y)**2)/(2*self.marker_size**2))
        # Marker and walls share the image channels; anti-aliasing preserves subpixel movement.
        clean = .16 + .39*self.background
        clean = clean*(1-marker)+.98*marker if not self.rgb else clean[None]*(1-marker)+np.array([.98,.08,.08])[:,None,None]*marker
        if self.noise_mode=='panel':
            image=clean.copy()
            panel=self.rng.uniform(0,1,clean.shape)
            image[...,self.viewport:,:]=panel[...,self.viewport:,:]
            image[...,:,self.viewport:]=panel[...,:,self.viewport:]
        else:
            image = clean + self.rng.normal(0,self.noise,clean.shape)
            if self.noise: image += self.rng.normal(0,self.noise*.3)
        pixels=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
        return pixels if self.rgb else pixels[None]


class PixelEncoder(nn.Module):
    def __init__(self, size=48, dim=8, architecture='cnn', width=128, center=False, channels=1, temperature=10.):
        super().__init__()
        self.architecture = architecture
        self.center=center
        # Historical CLI name: this is logit SHARPNESS (inverse temperature).
        self.temperature=temperature
        self.register_buffer("background",torch.zeros(1,channels,size,size))
        if architecture == 'mlp':
            self.net = nn.Sequential(nn.AvgPool2d(2),nn.Flatten(),mlp(channels*(size//2)**2,width,dim,3))
        elif architecture in {'spatial','keypoint'}:
            self.features = nn.Sequential(nn.Conv2d(channels,8,5,1,2),nn.SiLU(),nn.AvgPool2d(2),
                nn.Conv2d(8,8,3,1,1),nn.SiLU(),nn.Conv2d(8,1 if architecture=='keypoint' else 8,1))
            self.net = nn.Identity() if architecture=='keypoint' else mlp(16,width,dim,2)
            coords=torch.linspace(-1,1,size//2)
            yy,xx=torch.meshgrid(coords,coords,indexing='ij')
            self.register_buffer('grid',torch.stack([xx.flatten(),yy.flatten()],-1))
        else:
            self.features = nn.Sequential(nn.Conv2d(channels,8,5,2,2),nn.SiLU(),
                nn.Conv2d(8,16,3,2,1),nn.SiLU())
            self.net = nn.Sequential(nn.Flatten(),mlp(16*((size+3)//4)**2,width,dim,2))

    def forward(self,x):
        x=x.float()/255.-.5-self.background
        if self.architecture in {'spatial','keypoint'} or self.center:x=x-x.mean((-2,-1),keepdim=True)
        if self.architecture in {'spatial','keypoint'}:
            features=self.features(x).flatten(2)
            points=(features.mul(self.temperature).softmax(-1) @ self.grid).flatten(1)
            return self.net(points)
        return self.net(self.features(x) if self.architecture!='mlp' else x)


class PixelLearner:
    def __init__(self, args):
        self.encoder=PixelEncoder(args.image_size,args.dim,args.architecture,args.width,getattr(args,'center',False),3 if getattr(args,'rgb',False) else 1,getattr(args,'temperature',10.))
        self.heads=nn.ModuleList([mlp(args.dim+2,args.width,args.dim,3) for _ in range(3)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight,std=.01);nn.init.zeros_(head[-1].bias)
        self.inverse=mlp(args.dim*2,args.width,2,2) if getattr(args,'inverse',0) else None
        self.device=torch.device(getattr(args,'device','cpu'))
        self.encoder.to(self.device);self.heads.to(self.device)
        if self.inverse is not None:self.inverse.to(self.device)
        self.cpu_encoder=copy.deepcopy(self.encoder).cpu() if self.device.type!='cpu' else self.encoder
        self.cpu_version=-1
        self.optimizer=torch.optim.Adam(list(self.encoder.parameters())+list(self.heads.parameters())+(list(self.inverse.parameters()) if self.inverse else []),lr=args.lr)
        self.a=args; self.updates=0

    @torch.no_grad()
    def encode(self,images):
        if self.device.type!='cpu' and len(images)<32:
            if self.cpu_version!=self.updates:
                self.cpu_encoder.load_state_dict({k:v.detach().cpu() for k,v in self.encoder.state_dict().items()});self.cpu_version=self.updates
            z=self.cpu_encoder(torch.as_tensor(images))
        else:z=self.encoder(torch.as_tensor(images,device=self.device))
        if getattr(self.a,'normalize',0):z=torch.nn.functional.normalize(z,dim=-1)
        return z.cpu().numpy()

    def update(self,images,parents,actions,ids,rng,metrics=True):
        a=self.a
        # ids are child node IDs; every edge keeps the observed source image.
        negatives=rng.integers(0,len(images),size=(2,len(ids)))
        indices=np.concatenate([parents[ids],ids,negatives.ravel()])
        z,t,left,right=self.encoder(torch.as_tensor(images[indices],device=self.device)).chunk(4)
        controls=torch.as_tensor(actions[ids],device=self.device)
        pred=torch.stack([z+head(torch.cat([z,controls],-1)) for head in self.heads])
        dynamics=(pred-t).square().mean()
        adjacency=(z-t).square().sum(-1).mean()
        repulsion=(a.margin**2-(left-right).square().sum(-1)).clamp_min(0).mean()
        norm=torch.cat([left,right]).square().sum(-1).mean()
        states=torch.cat([left,right])
        std=(states.var(0,unbiased=False)+1e-6).sqrt()
        variance=(.1-std).clamp_min(0).mean()
        centered=states-states.mean(0)
        cov=centered.T @ centered/max(1,len(states)-1)
        decorrelation=(cov-torch.diag(cov.diag())).square().sum()
        loss=a.dynamics*dynamics+a.adjacency*adjacency+a.repulsion*repulsion+a.l2*norm
        loss=loss+getattr(a,'variance',0.)*variance+getattr(a,'covariance',0.)*decorrelation
        if getattr(a,'infonce',0):
            from torch.nn import functional as F
            logits=F.normalize(pred,dim=-1) @ F.normalize(t,dim=-1).T / .1
            contrast=F.cross_entropy(logits.flatten(0,1),torch.arange(len(t),device=self.device).repeat(len(self.heads)))
            loss=loss+a.infonce*contrast
        if getattr(a,'augmentation',0):
            raw=torch.as_tensor(images[negatives[0]],device=self.device).float()
            augmented=(raw+torch.randn_like(raw)*(.12*255)).clamp(0,255)
            augmented_z=self.encoder(augmented)
            loss=loss+a.augmentation*(augmented_z-left).square().sum(-1).mean()
        if self.inverse is not None:loss=loss+a.inverse*(self.inverse(torch.cat([z,t],-1))-controls).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.heads.parameters()),10)
        self.optimizer.step();self.updates+=1
        stats=dict(loss=loss.detach(),dynamics=dynamics.detach(),adjacency=adjacency.detach(),
                   repulsion=repulsion.detach(),norm=norm.detach(),variance=variance.detach())
        return {k:float(v) for k,v in stats.items()} if metrics else stats
