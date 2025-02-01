import unittest
import pandas as pd
from llm_pipeline.pipeline import LLMCallStep, FixedProcessingStep, DataPipeline, FilterStep
from data_preparation.documents import Document
from data_preparation.version_one_backlog_item import VersionOneBacklogItem
from typing import List


class TestLLMPipeline(unittest.TestCase):

    def test_pipeline(self):
        # Load test data from CSV file into a DataFrame
        try:
            df = pd.read_csv('./tests/test_data_pipeline.csv')
        except pd.errors.ParserError as e:
            print(f"Error reading CSV file: {e}")
            raise

        # Define a couple of processing steps.

        # Example fixed processing step: add a new column that shows the word count of the description.
        def count_words(df: pd.DataFrame) -> pd.DataFrame:
            df['word_count'] = df['description'].apply(lambda x: len(x.split()))
            return df

        fixed_step = FixedProcessingStep(count_words)

        # Example LLM processing step: use an LLM to extract potential UI change details.
        llm_step = LLMCallStep(
            prompt_template="""Determine if the following development story likely resulted in a UI update or not.
            respond back with only 'yes' or 'no'.

            Title: {title}
            Description: {description}
            Acceptance Criteria: {acceptance_criteria}""",
            output_key="ui_change",
            fields=["title", "description", "acceptance_criteria"]
        )

        # Example filter step: filter out rows that are not UI edits.
        def filter_function(df: pd.DataFrame) -> pd.Series:
            return df['ui_change'].str.strip().str.lower() == "yes"

        filter_step = FilterStep(filter_function)

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[fixed_step, llm_step, filter_step], batch_size=5)

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(df)

        # Print the results.
        print(len(processed_df))
        print(processed_df.to_string())

if __name__ == '__main__':
    unittest.main()