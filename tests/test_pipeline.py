import unittest
from llm_pipeline.process_pipeline import LLMPipeline

class TestLLMPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = LLMPipeline()

    def test_define_steps(self):
        steps = ['step1', 'step2', 'step3']
        self.pipeline.define_steps(steps)
        self.assertEqual(self.pipeline.steps, steps)

    def test_process_batches(self):
        self.pipeline.define_steps(['step1', 'step2'])
        data = ['data1', 'data2']
        processed_data = self.pipeline.process_batches(data)
        self.assertEqual(len(processed_data), len(data))

if __name__ == '__main__':
    unittest.main()