import unittest
from llm_pipeline.evaluate_batch import BatchEvaluator

class TestBatchEvaluator(unittest.TestCase):
    def test_classify_batch(self):
        system_prompt = "Please classify the following as either an animal, vegetable, or mineral.  Return only a single word, and nothing else"
        records = [
            "bear",
            "sugar cane",
            "quartz"
            ]
        evaluator = BatchEvaluator(system_prompt)

        results = evaluator.process_batch(records)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], 'animal')
        self.assertEqual(results[1], 'vegetable')
        self.assertEqual(results[2], 'mineral')


    def test_nlp_batch(self):
        system_prompt = "Please Extract the subject an predicate from the following sentences.  Return only response in the format: Subject: <subject>, Predicate: <predicate>"
        records = [
            "Bob rode his bike to the store",
            "The brown dog chased the cat",
            "There is no more coffee in the pot"
            ]
        evaluator = BatchEvaluator(system_prompt)

        results = evaluator.process_batch(records)

        self.assertEqual(len(results), 3)
        print(results)



if __name__ == '__main__':
    unittest.main()
