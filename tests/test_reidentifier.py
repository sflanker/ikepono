import unittest

from ikepono.configuration import Configuration
from ikepono.reidentifier import Reidentifier


class ReidentifierTests(unittest.TestCase):

    def test_private_initializer(self):
        with self.assertRaises(Exception):
            Reidentifier(Configuration())

    def test_training_config(self):
        reidentifier = Reidentifier.for_training(Configuration("test_configuration.json"))
        self.assertIsNotNone(reidentifier)
        self.assertIsNotNone(reidentifier.train_dataset)
        self.assertIsNotNone(reidentifier.sampler)
        self.assertIsNotNone(reidentifier.vector_store)
        self.assertIsNotNone(reidentifier.model)
        self.assertIsNotNone(reidentifier.configuration)
        self.assertIsNotNone(reidentifier.dataset_device)
        self.assertIsNotNone(reidentifier.validation_dataset)

    def test_train(self):
        reidentifier = Reidentifier.for_training(Configuration("test_configuration.json"))
        training_time_in_seconds, best_mrr = reidentifier.train()
        self.assertTrue(training_time_in_seconds > 10)
        self.assertTrue(best_mrr > 0.0)

if __name__ == '__main__':
    unittest.main()
