import unittest
import numpy as np
import torch
from traffic import *
from train import Policy,load_circuit

class TrafficTests(unittest.TestCase):
    def test_conservation_and_nonnegative_queues(self):
        for seed in range(5):
            rng=np.random.default_rng(seed);s=initial_state();cap=capacity(.5)
            for a in make_arrivals(seed,1.35):
                step(s,rng.integers(0,2,3),a,cap)
                self.assertAlmostEqual(s.q.sum()+s.departed,s.arrivals)
                self.assertTrue((s.q>=0).all())

    def test_clearance_and_minimum_green(self):
        s=initial_state();s.q[:]=10;s.arrivals=60
        flow,changed=step(s,[1,1,1],np.zeros((3,2)),capacity())
        self.assertTrue(changed.all());self.assertEqual(flow.sum(),0)
        for _ in range(2):
            _,changed=step(s,[0,0,0],np.zeros((3,2)),capacity())
            self.assertFalse(changed.any());self.assertTrue((s.phase==1).all())

    def test_spillback_blocks_upstream_without_losing_vehicles(self):
        s=initial_state();s.q[0,0]=20;s.q[1,0]=60;s.arrivals=80
        flow,_=step(s,[0,0,0],np.zeros((3,2)),capacity())
        self.assertEqual(flow[0,0],0);self.assertEqual(s.q.sum()+s.departed,80)

    def test_identical_arrivals_across_controllers_and_split_seeds(self):
        np.testing.assert_array_equal(make_arrivals(5100),make_arrivals(5100))
        self.assertFalse(np.array_equal(make_arrivals(5100),make_arrivals(5101)))
        self.assertEqual(make_arrivals(5100)[180:].sum(),0)

    def test_model_has_no_shortcut_around_output_neurons(self):
        torch.set_num_threads(2);n,e=load_circuit();model=Policy(n,e)
        x=torch.randn(4,15);mask=torch.zeros(len(n),dtype=torch.bool);mask[model.outputs]=True
        pred=model(x,ablate=mask)
        torch.testing.assert_close(pred,model.decoder.bias.expand_as(pred))

if __name__=='__main__':unittest.main(verbosity=2)
