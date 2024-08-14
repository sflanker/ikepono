import unittest
import torch
from pytorch_metric_learning import losses


class TestPTML(unittest.TestCase):
    def test_SubcenterArcFaceLoss(self):
        embedding_size = 128
        batch_size = 8
        numclasses = 10
        loss = losses.SubCenterArcFaceLoss(
            num_classes = numclasses,
            embedding_size = embedding_size
        )
        vs =  torch.rand(batch_size, embedding_size)
        labels = torch.randint(0, numclasses, (batch_size,))
        assert vs.shape == (batch_size, embedding_size), f"Expected shape {batch_size, embedding_size} got {vs.shape}"
        assert labels.shape == (batch_size,), f"Expected shape {batch_size} got {labels}"
        l = loss(vs, labels)
        assert 0.0 < l.item() < 100.0, f"Expected loss got {l.item()}"

if __name__ == '__main__':
    unittest.main()
