"""Focused checks for AdaMAL's alignment geometries and checkpoint policy."""

from __future__ import annotations

import copy
import itertools
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optims.mal_opt import AdaMAL


class AdaMALAlignmentChecks(unittest.TestCase):
    def test_alignment_uses_the_current_probe_and_correct_scaling_norm(self):
        # h_old=(2,-1), g=(1,1), beta1=.5 => h_probe=(2,.5).
        # v_old=(1,31), beta2=.5 => v_next=(1,16), distinctly anisotropic.
        cases = itertools.product(("moment", "update", "metric"), (False, True), (False, True),
                                  ("none", "moment", "step"), (.5, 1.), ("attenuate", "replace"))
        for align, unbias, recursive, scale, power, gate in cases:
            with self.subTest(align=align, unbias=unbias, recursive=recursive, scale=scale, power=power, gate=gate):
                eps = 1e-8
                p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
                opt = AdaMAL([p], lr=1., betas=(.5, .5), eps=eps, align=align,
                             unbias=unbias, in_place=recursive, scale=scale, pwr=power, gate_mode=gate)
                opt.state[p].update(step=3, momentum_buffer=torch.tensor([2., -1.], dtype=p.dtype),
                                    exp_avg_sq=torch.tensor([1., 31.], dtype=p.dtype))
                correction = 1-.5**4 if unbias else 1.
                d1, d2 = math.sqrt(1/correction)+eps, math.sqrt(16/correction)+eps
                if align == "moment":
                    grad = (1., 1.)
                    probe = (2., .5)
                elif align == "update":
                    grad = (1., 1.)
                    probe = (2/d1, .5/d2)
                else:
                    grad = (1/math.sqrt(d1), 1/math.sqrt(d2))
                    probe = (2/math.sqrt(d1), .5/math.sqrt(d2))
                a_norm = math.hypot(*grad)
                b_norm = math.hypot(*probe)
                cosine = sum(a*b for a, b in zip(grad, probe, strict=True))/(a_norm*b_norm)
                c = ((1+cosine)/2)**power * (.5 if gate == "attenuate" else 1.)
                h = torch.tensor([1+2*c, 1-c], dtype=p.dtype)
                D = torch.tensor([d1, d2], dtype=p.dtype)
                if scale == "moment":
                    h *= math.hypot(2., .5)/(h.norm()+eps)
                update = h/D
                if scale == "step":
                    update *= math.hypot(2/d1, .5/d2)/(update.norm()+eps)
                p.grad = torch.ones_like(p)
                opt.step()
                torch.testing.assert_close(-p, update, rtol=1e-12, atol=1e-12)
                expected_h = h if recursive else torch.tensor([2., .5], dtype=p.dtype)
                torch.testing.assert_close(opt.state[p]["momentum_buffer"], expected_h, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(opt.state[p]["exp_avg_sq"], torch.tensor([1., 16.], dtype=p.dtype))
                torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)

    def test_checkpoint_restores_alignment_and_continues_identically(self):
        for align, unbias, recursive, scale in itertools.product(("moment", "update", "metric"), (False, True), (False, True), ("none", "moment", "step")):
            with self.subTest(align=align, unbias=unbias, recursive=recursive, scale=scale):
                params = [torch.nn.Parameter(torch.ones(shape, dtype=torch.float64)) for shape in ((2, 2), (2,))]
                opt = AdaMAL(params, lr=.03, betas=(.8, .9), weight_decay=.1, align=align,
                             unbias=unbias, in_place=recursive, scale=scale)
                for p in params:
                    p.grad = torch.arange(1, p.numel()+1, dtype=p.dtype).reshape_as(p)
                opt.step()
                saved = copy.deepcopy(opt.state_dict())
                restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
                other_align = "metric" if align != "metric" else "moment"
                restored = AdaMAL(restored_params, weight_decay=.1, align=other_align, unbias=not unbias)
                restored.load_state_dict(saved)
                self.assertTrue(all(group["align"] == align for group in restored.param_groups))
                for p, r in zip(params, restored_params, strict=True):
                    p.grad = -.2*torch.ones_like(p)
                    r.grad = p.grad.clone()
                opt.step()
                restored.step()
                for p, r in zip(params, restored_params, strict=True):
                    torch.testing.assert_close(p, r, rtol=0, atol=0)
                    for key in ("momentum_buffer", "exp_avg_sq"):
                        torch.testing.assert_close(opt.state[p][key], restored.state[r][key], rtol=0, atol=0)

    def test_pre_alignment_checkpoint_keeps_historical_moment_mode(self):
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = AdaMAL([p], unbias=True)
        self.assertEqual(opt.param_groups[0]["align"], "moment")
        p.grad = torch.tensor([1., 3.], dtype=p.dtype)
        opt.step()
        saved = copy.deepcopy(opt.state_dict())
        for group in saved["param_groups"]:
            group.pop("align")
        r = torch.nn.Parameter(p.detach().clone())
        restored = AdaMAL([r], align="update")
        restored.load_state_dict(saved)
        self.assertEqual(restored.param_groups[0]["align"], "moment")
        self.assertNotIn("align", saved["param_groups"][0])
        p.grad = torch.tensor([.5, -1.], dtype=p.dtype)
        r.grad = p.grad.clone()
        opt.step()
        restored.step()
        torch.testing.assert_close(p, r, rtol=0, atol=0)

    def test_only_supported_alignment_modes_are_accepted(self):
        for align in ("white", "unknown", None):
            with self.subTest(align=align):
                p = torch.nn.Parameter(torch.ones(2))
                with self.assertRaisesRegex(ValueError, "align"):
                    AdaMAL([p], align=align)
                saved = copy.deepcopy(AdaMAL([p]).state_dict())
                saved["param_groups"][0]["align"] = align
                with self.assertRaisesRegex(ValueError, "align"):
                    AdaMAL([p]).load_state_dict(saved)

    def test_zero_gradient_uses_base_memory_coefficient_in_both_modes(self):
        for align in ("moment", "update", "metric"):
            p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            opt = AdaMAL([p], align=align, in_place=True)
            p.grad = torch.tensor([1., -2.], dtype=p.dtype)
            opt.step()
            before = opt.state[p]["momentum_buffer"].clone()
            p.grad = torch.zeros_like(p)
            opt.step()
            torch.testing.assert_close(opt.state[p]["momentum_buffer"], .9*before, rtol=1e-12, atol=1e-12)
            self.assertTrue(torch.isfinite(p).all())

    def test_latest_step_diagnostics_are_parameter_count_weighted(self):
        first = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        second = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        opt = AdaMAL((first, second), betas=(.8, .9), align="moment")
        opt.state[first].update(
            step=1,
            momentum_buffer=torch.tensor([-2., 0.], dtype=first.dtype),
            exp_avg_sq=torch.ones_like(first),
        )
        opt.state[second].update(
            step=1,
            momentum_buffer=torch.zeros_like(second),
            exp_avg_sq=torch.ones_like(second),
        )
        first.grad = torch.tensor([1., 0.], dtype=first.dtype)
        second.grad = torch.ones_like(second)
        opt.step()

        first_probe = first.grad + .8 * torch.tensor([-2., 0.], dtype=first.dtype)
        first_cosine = torch.dot(first.grad, first_probe) / (first.grad.norm() * first_probe.norm())
        first_gate = (1. + first_cosine) / 2.
        expected_gate = (2 * first_gate + 6.) / 8.
        torch.testing.assert_close(opt.last_gate_mean, expected_gate)
        torch.testing.assert_close(opt.last_gate_min, first_gate)
        torch.testing.assert_close(opt.last_gate_max, torch.ones_like(first_gate))
        torch.testing.assert_close(opt.last_beta_eff_mean, .8 * expected_gate)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_both_alignment_modes_match_cpu_on_mps(self):
        for align, unbias, recursive, scale in itertools.product(("moment", "update", "metric"), (False, True), (False, True), ("none", "moment", "step")):
            with self.subTest(align=align, unbias=unbias, recursive=recursive, scale=scale):
                cpu = torch.nn.Parameter(torch.ones((2, 2)))
                mps = torch.nn.Parameter(cpu.detach().to("mps"))
                opts = [AdaMAL([p], lr=.03, weight_decay=.1, eps=1e-7, align=align,
                               unbias=unbias, in_place=recursive, scale=scale) for p in (cpu, mps)]
                for values in ([1., 2., -.5, .3], [-.4, .1, .5, -.2], [0., 0., 0., 0.]):
                    g = torch.tensor(values).reshape(2, 2)
                    cpu.grad, mps.grad = g.clone(), g.to("mps")
                    for opt in opts:
                        opt.step()
                    torch.testing.assert_close(mps.detach().cpu(), cpu, rtol=2e-5, atol=2e-6)
                    for key in ("momentum_buffer", "exp_avg_sq"):
                        torch.testing.assert_close(opts[1].state[mps][key].cpu(), opts[0].state[cpu][key], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
