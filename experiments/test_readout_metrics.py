"""CPU tests of readout statistics using an independent NumPy tensor adapter."""
import unittest
import numpy as np
try:
    from .readout_metrics import measure_readouts, summarize_readouts, select_positions
except ImportError:
    from readout_metrics import measure_readouts, summarize_readouts, select_positions


class Tensor:
    def __init__(self, data): self.a = np.asarray(data)
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    def float(self): return self
    def __iter__(self): return (Tensor(x) for x in self.a)
    def __getitem__(self, index):
        if isinstance(index, Tensor): index = index.item()
        return Tensor(self.a[index])
    def __gt__(self, other): return Tensor(self.a > other.a)
    def __sub__(self, other): return Tensor(self.a - other.a)
    def sum(self): return Tensor(self.a.sum())
    def item(self): return self.a.item()
    def tolist(self): return self.a.tolist()
    def isfinite(self): return Tensor(np.isfinite(self.a))
    def all(self): return Tensor(self.a.all())
    def argmax(self, dim): return Tensor(self.a.argmax(axis=dim))
    def softmax(self, dim):
        x = self.a - self.a.max(axis=dim, keepdims=True)
        exp = np.exp(x)
        return Tensor(exp/exp.sum(axis=dim, keepdims=True))
    def topk(self, k, dim):
        ids = np.argsort(-self.a, axis=dim, kind='stable')[..., :k]
        return Tensor(np.take_along_axis(self.a, ids, axis=dim)), Tensor(ids)


class ReadoutTests(unittest.TestCase):
    def test_rank_and_membership_are_not_only_top1(self):
        early = Tensor([[4, 3, 2, 1, 0, -1], [4, 3, 2, 1, 0, -1]])
        final = Tensor([[0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 5]])
        rows = measure_readouts(early, final, [2,5])
        self.assertEqual([r['final_rank_in_early'] for r in rows], [3,6])
        self.assertEqual([r['final_in_early_top5'] for r in rows], [True,False])
        self.assertTrue(all(r['final_in_early_top10'] for r in rows))
        self.assertGreater(rows[0]['final_confidence'], rows[0]['final_probability_in_early'])

    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            measure_readouts(Tensor([[float('nan'),0]]), Tensor([[1,0]]), [0])

    def test_single_vocab(self):
        row = measure_readouts(Tensor([[1]]), Tensor([[1]]), [0])[0]
        self.assertEqual(row['early_probability_margin'], 1)
        self.assertEqual(row['final_rank_in_early'], 1)

    def test_summary_separates_eos_content_and_first(self):
        rows = measure_readouts(Tensor([[4,0],[4,0]]), Tensor([[4,0],[0,4]]), [0,1])
        for i, row in enumerate(rows):
            row.update(id='s',stage='H1',position=i,is_eos=i==1)
        result = summarize_readouts(rows)['H1']
        self.assertEqual(result['all']['compared_tokens'], 2)
        self.assertEqual(result['all']['top1_agreement'], .5)
        self.assertEqual(result['content']['top1_agreement'], 1)
        self.assertEqual(result['eos']['top1_agreement'], 0)
        self.assertEqual(result['first']['compared_tokens'], 1)

    def test_checks_include_endpoints_without_duplicates(self):
        self.assertEqual(select_positions(4, 8), [0,1,2,3])
        self.assertEqual(select_positions(10, 4), [0,3,6,9])
        self.assertEqual(select_positions(0, 4), [])
        self.assertEqual(select_positions(10, 0), [])


if __name__ == '__main__':
    unittest.main()
