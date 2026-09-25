"""任务损失的数值与梯度检查。"""
import unittest
import torch
from torch.nn import functional as F
from scripts.train import supervised_loss


class TrainingObjectivesTest(unittest.TestCase):
    def test_l1_matches_absolute_error(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        prediction = torch.tensor([0.2, -0.4], requires_grad=True)
        batch = {"classes": torch.tensor([1, 0]), "sentiment": torch.tensor([0., -2.])}
        output = {"logits": logits, "sentiment": prediction}
        loss = supervised_loss(output, batch, regression_loss="l1")
        self.assertTrue(torch.allclose(loss, F.cross_entropy(logits, batch["classes"]) + torch.tensor(0.9)))
        loss.backward()
        self.assertTrue(torch.allclose(prediction.grad, torch.tensor([0.5, 0.5])))

    def test_default_keeps_previous_objective(self):
        output = {"logits": torch.zeros(2, 3), "sentiment": torch.tensor([0.2, -0.4])}
        batch = {"classes": torch.tensor([1, 0]), "sentiment": torch.tensor([0., -2.])}
        expected = F.cross_entropy(output["logits"], batch["classes"]) + F.smooth_l1_loss(output["sentiment"], batch["sentiment"])
        self.assertTrue(torch.equal(supervised_loss(output, batch), expected))
