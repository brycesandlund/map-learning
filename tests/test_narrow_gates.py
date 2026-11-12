import numpy as np
import pytest
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze

@pytest.mark.parametrize('width',[1.,.25,.0625,.015625])
def test_gates_are_passable_but_walls_block(width):
    e=PixelMaze(MazeConfig(),family='gates',gate_width=width)
    for i in range(1,4):
        x=i*3;y=1.73 if i%2 else 10.27
        for sign in [-1,1]:
            for offset in [-.49*width,0,.49*width]:
                p=np.array([x-sign*.1,y+offset],np.float32)
                q=e.transition(p,np.array([sign,0],np.float32))
                assert (q[0]-x)*sign>0
            for offset in [-.51*width,.51*width]:
                p=np.array([x-sign*.1,y+offset],np.float32)
                assert np.array_equal(p,e.transition(p,np.array([sign,0],np.float32)))
    # Construct a complete route through all three entrances without any resets.
    p=e.start.copy()
    for i in range(1,4):
        for target in [[(i-1)*3+1.5,1.73 if i%2 else 10.27],[i*3+1.5,1.73 if i%2 else 10.27]]:
            target=np.array(target,np.float32)
            for axis in range(2):
                while abs(target[axis]-p[axis])>1e-5:
                    delta=float(target[axis]-p[axis]);sign=np.sign(delta)
                    # A small net shift uses two legal, opposing movements.
                    lengths=[min(.25,abs(delta))] if abs(delta)>=.125 else [.125+abs(delta),-.125]
                    for length in lengths:
                        control=np.zeros(2,np.float32);control[axis]=sign*length/.25
                        assert .5-1e-6<=abs(control[axis])<=1+1e-6
                        q=e.transition(p,control)
                        assert np.linalg.norm(q-p)>1e-6
                        p=q
    assert p[0]>9


def test_subpixel_opening_not_rounded_shut():
    images=[PixelMaze(MazeConfig(),family='gates',gate_width=w).background for w in [.015625,.0625,.25,1.]]
    for a,b in zip(images,images[1:]):
        assert np.all(b>=a-1e-6)
        assert np.sum(b-a)>.01
