import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

##############################################################################
# 2. Pipeline Step Abstraction
##############################################################################

class PipelineStep(ABC):
    """
    Abstract base class for a pipeline processing step.
    Each step should implement the process() method.
    """

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame and return the updated DataFrame.
        """
        pass

##############################################################################
# 3. Example Pipeline Steps
##############################################################################

class LLMCallStep(PipelineStep):
    """
    A processing step that builds a prompt (using the DataFrame’s data),
    calls an LLM, and stores the response in the DataFrame.
    """
    def __init__(self, prompt_template: str, output_key: str = "llm_output", fields: List[str] = None):
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.fields = fields
        self.system_prompt = """You are evaluating records. The user will submit a record to be evaluated, and you should respond with only the evaluation. 
        Respond only with the evaluation, and no other terms or formatting.
        
        The requested evaluation is:"""
        self.client = OpenAI()
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Generate the record details for each row using the provided fields or all fields.
        def create_record_details(row):
            if self.fields:
                details = {field: row[field] for field in self.fields}
            else:
                details = row.to_dict()
            return "\n".join([f"{key}: {value}" for key, value in details.items()])
        
        df.loc[:, 'record_details'] = df.apply(create_record_details, axis=1)
        
        # Generate the prompt for each row using the record details.
        df.loc[:, 'prompt'] = df['record_details'].apply(
            lambda details: self.prompt_template.format(record_details=details)
        )
        
        # Define a helper function to call the LLM for each prompt.
        def get_llm_response(prompt: str) -> str:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            # Return the LLM's response.
            return chat_completion.choices[0].message.content.strip()
        
        # Use the helper function with .apply() to get the output.
        df.loc[:, self.output_key] = df['prompt'].apply(get_llm_response)
        
        # Optionally, drop the temporary columns.
        df.drop(columns=['record_details', 'prompt'], inplace=True)
        
        return df

class FixedProcessingStep(PipelineStep):
    """
    A processing step that uses a fixed function to process the DataFrame.
    """
    def __init__(self, process_function: Callable[[pd.DataFrame], pd.Series], output_key: str):
        self.process_function = process_function
        self.output_key = output_key

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.output_key] = self.process_function(df)
        return df

class FilterStep(PipelineStep):
    """
    A processing step that filters rows in the DataFrame based on a condition.
    """
    def __init__(self, filter_function: Callable[[pd.DataFrame], pd.Series]):
        self.filter_function = filter_function

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.filter_function(df)]

##############################################################################
# 4. The Pipeline: Running Steps in Batches
##############################################################################

class DataPipeline:
    """
    A generic pipeline to process data in batches.
    The pipeline runs each batch through each of the configured steps.
    """
    def __init__(self, steps: List[PipelineStep], batch_size: int = 100):
        self.steps = steps
        self.batch_size = batch_size

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = pd.DataFrame()
        for batch in self._batch_data(df, self.batch_size):
            processed_batch = self._process_batch(batch)
            processed_df = pd.concat([processed_df, processed_batch], ignore_index=True)
        return processed_df

    def _batch_data(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        # Create a copy of each slice to avoid SettingWithCopyWarning when modifying the batch.
        return [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]

    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            batch = step.process(batch)
        return batch