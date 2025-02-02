import unittest
import pandas as pd
from llm_pipeline.pipeline import LLMCallStep, FixedProcessingStep, DataPipeline, FilterStep, GenerateEmbeddingsStep, kNNFilterStep, LLMCallWithDataFrame
import csv
import csv
from typing import List


class TestLLMPipeline(unittest.TestCase):

    def test_classify_pipeline(self):
        # Load test data from CSV file into a DataFrame
        try:
            df = pd.read_csv('./tests/test_data_pipeline.csv', quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError as e:
            print(f"Error reading CSV file: {e}")
            raise
        except FileNotFoundError as e:
            print(f"CSV file not found: {e}")
            raise

        # Define a couple of processing steps.

        # Example fixed processing step: add a new column that shows the word count of the description.
        def count_words(df: pd.DataFrame) -> pd.Series:
            return df['description'].apply(lambda x: len(x.split()))

        fixed_step = FixedProcessingStep(count_words, output_key="word_count")

        # Example LLM processing step: use an LLM to extract potential UI change details.
        llm_step = LLMCallStep(
            prompt_template="""Determine if the following development story likely resulted in a UI update or not.
            respond back with only 'yes' or 'no'.

            {record_details}""",
            output_key="ui_change",
            fields=["title", "description", "acceptance_criteria"]
        )

        # Example filter step: filter out rows that are not UI edits.
        def filter_function(df: pd.DataFrame) -> pd.Series:
            return df['ui_change'].str.strip().str.lower() == "yes"

        filter_step = FilterStep(filter_function)

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[fixed_step, llm_step, filter_step])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(df)

        # Print the results.
        print(len(processed_df))
        print(processed_df[['id', 'title']].to_string())


    def test_gen_and_persist_embeddings(self):
        # Load test data from CSV file into a DataFrame
        try:
            df = pd.read_csv('./tests/test_data_pipeline.csv', quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError as e:
            print(f"Error reading CSV file: {e}")
            raise

        # Define a couple of processing steps.

        # Generate one embedding from title & description.
        gen_embed= GenerateEmbeddingsStep(
            output_key="embedding_title_desc",
            fields=["title", "description"]
        )

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[gen_embed])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(df)
        processed_df.to_pickle('./tests/test_data_pipeline_embeddings.pkl')


    def test_reload_embeddings_and_kNN(self):
        
        # Reload the DataFrame.
        reloaded_df = pd.read_pickle('./tests/test_data_pipeline_embeddings.pkl')

        # Use a kNN filter on one of the embeddings:
        knn_filter = kNNFilterStep(
            query="input form",
            k=1,
            embedding_column="embedding_title_desc"
        )

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[knn_filter])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(reloaded_df)

        print(processed_df[['id', 'title']].to_string())

        llm_eval = LLMCallWithDataFrame(
            prompt_template="""What UIs changes are being implemented?'.

            {record_details}""",
            fields=["title", "description", "acceptance_criteria"],
        )
        print(llm_eval.call_llm(processed_df))


if __name__ == '__main__':
    unittest.main()