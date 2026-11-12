import numpy as np
import torch
import pytest
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze,PixelEncoder,PixelLearner
from map_learning.maze import bfs

@pytest.mark.parametrize('family',['maze','snake','comb','open','rooms'])
def test_all_rooms_connected(family):
    e=PixelMaze(MazeConfig(size=6),123,family)
    pts=np.array([[x+.5,y+.5] for x in range(6) for y in range(6)],np.float32)
    neighbors=[]
    old=e.config.step_size;e.config.step_size=1.
    for p in pts:
        neighbors.append([np.argmin(np.linalg.norm(pts-e.transition(p,a),axis=1)) for a in range(5)])
    assert np.all(bfs(np.array(neighbors),0)>=0)
    e.config.step_size=old


def test_images_no_separate_marker_and_fresh_noise():
    e=PixelMaze(MazeConfig(),123)
    x=e.observe();y=e.observe()
    assert x.shape==(1,48,48) and x.dtype==np.uint8
    assert np.mean(x!=y)>.9
    e.noise=0
    x=e.observe();assert np.array_equal(x,e.observe())
    # Subpixel movement is visible rather than snapping to a pixel ID.
    assert not np.array_equal(x,e.observe(e.position+[.04,0]))

@pytest.mark.parametrize('architecture',['cnn','mlp','spatial','keypoint'])
def test_encoder_pixel_gradient(architecture):
    dim=2 if architecture=='keypoint' else 8
    model=PixelEncoder(48,dim,architecture)
    x=torch.randint(0,256,(4,1,48,48),dtype=torch.uint8)
    z=model(x);assert z.shape==(4,dim)
    z.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rgb_is_composited_and_noisy():
    e=PixelMaze(MazeConfig(),123,rgb=True,marker_size=1.3)
    x=e.observe();y=e.observe()
    assert x.shape==(3,48,48) and x.dtype==np.uint8
    assert all(np.mean(x[c]!=y[c])>.9 for c in range(3))
    model=PixelEncoder(48,8,'cnn',channels=3)
    assert model(torch.from_numpy(x[None])).shape==(1,8)


def test_noise_panel_does_not_occlude_maze():
    e=PixelMaze(MazeConfig(image_size=56),123,noise_mode='panel')
    x=e.observe();y=e.observe()
    assert np.array_equal(x[:,:48,:48],y[:,:48,:48])
    assert np.mean(x[:,48:,:]!=y[:,48:,:])>.95
    assert not np.array_equal(x,e.observe(e.position+[.2,0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_training_and_cpu_inference_copy():
    from types import SimpleNamespace
    from map_learning.training import seed_all
    base=dict(image_size=48,dim=2,architecture='spatial',width=128,lr=.0003,dynamics=1.,adjacency=0.,repulsion=.3,l2=.03,margin=.25,temperature=200,center=1,rgb=0)
    seed_all(501,1);cpu=PixelLearner(SimpleNamespace(**base,device='cpu'))
    seed_all(501,1);gpu=PixelLearner(SimpleNamespace(**base,device='mps'))
    rng=np.random.default_rng(2);images=rng.integers(0,256,(65,1,48,48),dtype=np.uint8);parents=np.r_[-1,np.arange(64)];actions=rng.uniform(-1,1,(65,2)).astype(np.float32);ids=np.arange(1,65)
    assert np.max(abs(cpu.encode(images)-gpu.encode(images)))<1e-5
    r=cpu.update(images,parents,actions,ids,np.random.default_rng(3));s=gpu.update(images,parents,actions,ids,np.random.default_rng(3))
    assert abs(r['loss']-s['loss'])<1e-5
    # Tiny CPU-copy inference must reflect current GPU weights, not initial weights.
    z=gpu.encode(images[:5]);expected=gpu.encoder(torch.from_numpy(images[:5]).to('mps')).detach().cpu().numpy()
    assert np.max(abs(z-expected))<1e-5
    
