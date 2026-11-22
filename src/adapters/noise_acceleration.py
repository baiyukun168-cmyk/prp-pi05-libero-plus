"""Opt-in deferred LIBERO+ noise. Original source files remain untouched."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from numba import njit
from PIL import Image
from skimage.filters import gaussian

PARAMETERS=((.5,1,3),(.7,1,3),(.9,2,3),(1.,2,2),(1.1,3,2),
            (1.3,3,2),(1.5,4,2),(1.8,4,2),(2.2,5,1),(2.5,5,1))
NAMES=('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')


def same_state(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


def rng_hash():
    state=np.random.get_state()
    return hashlib.sha256(state[1].tobytes()+repr(state[2:]).encode()).hexdigest()


@njit(cache=False)
def _shuffle(x,delta,iterations,offsets):
    index=0
    for _ in range(iterations):
        for h in range(x.shape[0]-delta,delta,-1):
            for w in range(x.shape[1]-delta,delta,-1):
                dx,dy=offsets[index,0],offsets[index,1];index+=1
                for channel in range(x.shape[2]):
                    x[h,w,channel]=x[h+dy,w+dx,channel]
    return x


def fast_glass(image,severity):
    sigma,delta,iterations=PARAMETERS[severity-1]
    x=np.uint8(gaussian(np.array(image)/255.,sigma=sigma,channel_axis=-1)*255)
    count=iterations*max(0,x.shape[0]-2*delta)*max(0,x.shape[1]-2*delta)
    offsets=np.random.randint(-delta,delta,size=(count,2))
    x=_shuffle(x,delta,iterations,offsets)
    return np.clip(gaussian(x/255.,sigma=sigma,channel_axis=-1),0,1)*255


class DeferredImage:
    def __init__(self,handle,function,image,severity,before,after,frame):
        self.handle=handle;self.function=function;self.image=image.copy()
        self.severity=severity;self.before=before;self.after=after;self.frame=frame
        self.casts=[];self.value=None

    def materialize(self):
        if self.value is None:
            current=np.random.get_state();start=time.perf_counter()
            try:
                np.random.set_state(self.before)
                value=self.function(self.image,self.severity)
                if not same_state(self.after,np.random.get_state()):
                    raise RuntimeError('Noise RNG equivalence failed')
            finally:
                np.random.set_state(current)
            for args,kwargs in self.casts:value=value.astype(*args,**kwargs)
            self.value=value;self.handle.stats['computed']+=1
            self.handle.log('materialized',self.frame,image_hash=hashlib.sha256(np.asarray(value,dtype=np.uint8).tobytes()).hexdigest(),compute_ms=(time.perf_counter()-start)*1000)
        return self.value

    def astype(self,*args,**kwargs):
        if self.value is not None:return self.value.astype(*args,**kwargs)
        self.casts.append((args,kwargs));return self

    def __array__(self,dtype=None,copy=None):
        value=np.asarray(self.materialize(),dtype=dtype)
        return value.copy() if copy else value

    def __getitem__(self,key):return self.materialize()[key]
    def __len__(self):return len(self.materialize())
    def __getattr__(self,name):return getattr(self.materialize(),name)


class NoiseHandle:
    def __init__(self,wrapper,mode,audit_path):
        self.wrapper=wrapper;self.mode=mode
        self.originals={n:getattr(wrapper,n) for n in NAMES}
        self.stats={'created':0,'computed':0};self.episode=None;self.frame=0
        self.audit=Path(audit_path).open('x',buffering=1) if audit_path else None

    def log(self,event,frame=None,**values):
        if self.audit:
            self.audit.write(json.dumps({'event':event,'episode':self.episode,'frame':frame,**values})+'\n')

    def context(self,episode):
        self.episode=episode;self.frame=0

    def advance(self,group,severity,image):
        if group==0:np.random.uniform(-45,45)
        elif group==3:
            decay=(3,2.8,2.5,2.2,2.,1.8,1.6,1.5,1.4,1.3)[severity-1]
            self.wrapper.plasma_fractal(wibbledecay=decay)
        elif group==4:
            _,delta,iterations=PARAMETERS[severity-1]
            width,height=image.size
            count=iterations*max(0,height-2*delta)*max(0,width-2*delta)
            np.random.randint(-delta,delta,size=(count,2))

    def make(self,group,name):
        original=self.originals[name]
        def wrapped(image,severity=1):
            self.frame+=1;frame=self.frame;self.stats['created']+=1
            if self.mode=='original':
                start=time.perf_counter();value=original(image,severity)
                self.stats['computed']+=1
                self.log('created',frame,noise=name,severity=severity,rng_hash=rng_hash())
                self.log('materialized',frame,image_hash=hashlib.sha256(np.asarray(value,dtype=np.uint8).tobytes()).hexdigest(),compute_ms=(time.perf_counter()-start)*1000)
                return value
            before=np.random.get_state();self.advance(group,severity,image);after=np.random.get_state()
            self.log('created',frame,noise=name,severity=severity,rng_hash=rng_hash())
            function=fast_glass if group==4 else original
            return DeferredImage(self,function,image,severity,before,after,frame)
        return wrapped

    def restore(self):
        for name,function in self.originals.items():setattr(self.wrapper,name,function)
        if self.audit:
            self.log('summary',**self.stats);self.audit.close();self.audit=None


def install_noise_acceleration(wrapper,*,expected_source_sha256,mode='lazy',audit_path=None):
    if mode not in ('lazy','original'):raise ValueError('Unknown noise mode')
    source=Path(wrapper.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected_source_sha256:
        raise RuntimeError('Upstream noise source does not match validated version')
    state=np.random.get_state()
    try:fast_glass(Image.fromarray(np.zeros((16,16,3),np.uint8)),1)
    finally:np.random.set_state(state)
    handle=NoiseHandle(wrapper,mode,audit_path)
    for group,name in enumerate(NAMES):setattr(wrapper,name,handle.make(group,name))
    atexit.register(handle.restore)
    return handle
