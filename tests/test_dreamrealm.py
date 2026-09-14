import copy
import math
import tempfile
import unittest
import torch
from dreamrealm.objective import family_loss
from dreamrealm.engine import backward_family, CPUAdamW
from dreamrealm.prepare import validate_family, convert

class DreamRealmTest(unittest.TestCase):
    def test_noedit_boundaries(self):
        s = torch.tensor([[-1., -2.]], requires_grad=True)
        got, _ = family_loss(s, False, .7, .4, 99.)
        want = -s[0,0]+.4*torch.nn.functional.softplus((s[0,1]-s[0,0])/.7)
        torch.testing.assert_close(got, want)
        empty, terms = family_loss(s[:,:1], False)
        torch.testing.assert_close(empty, -s[0,0]); self.assertEqual(float(terms['row']),0)

    def test_permutation_and_column_signal(self):
        scores = torch.tensor([[-.1,-2.,-3.],[-3.,-.2,-1.],[-2.,-3.,-.3]])
        l,_ = family_loss(scores,True)
        perm=torch.tensor([2,0,1]);lp,_=family_loss(scores[perm][:,perm],True)
        torch.testing.assert_close(l,lp)
        uniform = torch.full((3,3), -1.)
        _,terms=family_loss(uniform,True)
        self.assertAlmostEqual(float(terms['row']), math.log(3), places=6)
        self.assertAlmostEqual(float(terms['col']), math.log(3), places=6)

    def test_recompute_matches_joint_autograd_variable_families(self):
        # Check full parameter gradients for variable N, mixed branches and family
        # weighting. This catches per-chunk softmax or per-cell normalization bugs.
        torch.manual_seed(81)
        for shapes in [[(2,2),(3,3),(1,4),(1,1)],[(16,16),(1,2)]]:
            direct=torch.nn.Linear(5,1); cached=copy.deepcopy(direct)
            xs=[torch.randn(n*m,5) for n,m in shapes]
            losses=[]
            for shape,x in zip(shapes,xs):
                losses.append(family_loss(direct(x).reshape(shape),shape[0]>1,.7,.4,.6)[0])
            torch.stack(losses).mean().backward()
            for shape,x in zip(shapes,xs):
                e=dict(family_id='test',shape=shape,with_edit=shape[0]>1,cells=list(x))
                backward_family(e,lambda cell:cached(cell).squeeze(),1/len(shapes),.7,.4,.6)
            for p,q in zip(direct.parameters(),cached.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)

    def test_cpu_optimizer_resume_matches_adamw(self):
        torch.manual_seed(2)
        a=torch.nn.Linear(3,2);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01)
        ob=CPUAdamW(b,.001,.01)
        x=torch.randn(2,3)
        for _ in range(2):
            oa.zero_grad();ob.zero_grad();a(x).square().mean().backward();b(x).square().mean().backward()
            oa.step();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
        with tempfile.TemporaryDirectory() as td:
            path=td+'/optimizer.pt';torch.save(ob.state_dict(),path)
            c=copy.deepcopy(b);oc=CPUAdamW(c,.001,.01);oc.load_state_dict(torch.load(path,weights_only=True))
            for model,opt in [(b,ob),(c,oc)]:opt.zero_grad();model(x).square().mean().backward();opt.step()
            for p,q in zip(b.parameters(),c.parameters()):torch.testing.assert_close(p,q)

    def test_invalid_family_is_rejected(self):
        f=dict(family_id='x',question='q',original_image='a',original_answer='same',edited_pairs=[dict(image='b',answer=' same ')],original_negative_answers=[])
        with self.assertRaisesRegex(ValueError,'duplicate_answers'):validate_family(f)

    def test_recompute_detects_stochastic_scores(self):
        calls=[0];p=torch.nn.Parameter(torch.ones(()))
        def score(_):calls[0]+=1;return p*calls[0]
        e=dict(family_id='x',shape=(1,1),with_edit=False,cells=[0])
        with self.assertRaisesRegex(RuntimeError,'drift'):backward_family(e,score)

if __name__=='__main__':unittest.main()
