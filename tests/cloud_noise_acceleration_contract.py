"""Cloud integration contract; stage beside the probe and adapter, then run explicitly."""
import ast
import importlib.util
from pathlib import Path
import hashlib
import os
import unittest
from types import SimpleNamespace

import probe_noise_acceleration as probe
import numpy as np
from PIL import Image

MODULE=Path(os.environ.get('PRP_NOISE_ADAPTER','noise_acceleration.py'))


class DeferredNoiseTest(unittest.TestCase):
    def test_consumed_pixels_and_every_rng_state(self):
        spec=importlib.util.spec_from_file_location('adapter_under_test',MODULE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        install=getattr(module,'install_noise_acceleration',None)
        self.assertTrue(callable(install),'deferred noise adapter is missing')
        definitions=[n for n in ast.parse(probe.SOURCE.read_text()).body
                     if isinstance(n,ast.FunctionDef) or isinstance(n,ast.ClassDef) and n.name=='MotionImage']
        scope=dict(vars(probe));scope['__file__']=str(probe.SOURCE)
        probe.wandlibrary.MagickMotionBlurImage.argtypes=(probe.ctypes.c_void_p,probe.ctypes.c_double,probe.ctypes.c_double,probe.ctypes.c_double)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(probe.SOURCE),'exec'),scope)
        wrapper=SimpleNamespace(**scope)
        originals={name:getattr(wrapper,name) for name in ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')}
        np.random.seed(818);before=np.random.get_state()
        handle=install(wrapper,expected_source_sha256=hashlib.sha256(probe.SOURCE.read_bytes()).hexdigest())
        self.assertTrue(probe.same_state(before,np.random.get_state()),'warmup changed RNG')
        image=Image.fromarray(np.random.RandomState(9).randint(0,256,size=(32,32,3),dtype=np.uint8))
        for group,name in enumerate(originals):
            for severity in range(1,11):
                np.random.seed(700+group*10+severity)
                expected=[];states=[]
                for step in range(5):
                    np.random.uniform(size=3)
                    a=np.asarray(originals[name](image,severity),dtype=np.uint8)
                    if step in (0,4):expected.append(a)
                    states.append(np.random.get_state())
                np.random.seed(700+group*10+severity)
                for step in range(5):
                    np.random.uniform(size=3)
                    a=getattr(wrapper,name)(image,severity).astype(np.uint8)
                    self.assertTrue(probe.same_state(states[step],np.random.get_state()),(name,severity,step))
                    if step in (0,4):
                        current=np.random.get_state()
                        self.assertTrue(np.array_equal(np.asarray(a),expected.pop(0)))
                        self.assertTrue(np.array_equal(a[::-1,::-1],np.asarray(a)[::-1,::-1]))
                        self.assertTrue(probe.same_state(current,np.random.get_state()),'reading advanced RNG')
        self.assertEqual(handle.stats['created'],250)
        self.assertEqual(handle.stats['computed'],100)
        handle.restore()
        self.assertIs(wrapper.glass_blur,originals['glass_blur'])


if __name__=='__main__':unittest.main()
