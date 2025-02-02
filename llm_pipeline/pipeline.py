import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, List
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

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
        self.system_prompt = (
            "You are evaluating records. The user will submit a record to be evaluated, and you should respond with only the evaluation. \n"
            "Respond only with the evaluation, and no other terms or formatting.\n\n"
            "The requested evaluation is:"
        )
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
# New Processor: GenerateEmbeddingsStep
##############################################################################


def openai_embedding_function(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using OpenAI's embeddings model.
    
    Args:
        text (str): The input text to embed.
        
    Returns:
        List[float]: The embedding vector.
    """
    client = OpenAI()
    try:
        response = client.embeddings.create(
            input=f"{text}",
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print("Error generating embedding:", e)
        return []


class GenerateEmbeddingsStep(PipelineStep):
    """
    A processing step that generates an embedding for each record using a supplied
    embedding function, and adds the result as a new column in the DataFrame.
    
    You can specify the output_key so that you can generate multiple embeddings.
    Persistence or caching should be handled outside this processor.
    """
    def __init__(self,
                 embedding_function: Callable[[str], List[float]] = openai_embedding_function,
                 output_key: str = "embedding",
                 fields: List[str] = None):
        """
        :param embedding_function: A function that takes a text string and returns an embedding vector (list of floats).
        :param output_key: Column name where the embedding will be stored.
        :param fields: List of DataFrame column names to combine for the embedding. If None, all columns are used.
        """
        self.embedding_function = embedding_function
        self.output_key = output_key
        self.fields = fields

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Define a helper to get the text for embedding.
        def get_text(row):
            if self.fields:
                return " ".join([str(row[field]) for field in self.fields])
            else:
                # Use all columns (converted to string)
                return " ".join(row.astype(str))
        
        # Compute the embedding for each row and store in the specified output_key.
        df.loc[:, self.output_key] = df.apply(lambda row: self.embedding_function(get_text(row)), axis=1)
        return df

##############################################################################
# New Processor: kNNFilterStep
##############################################################################

class kNNFilterStep(PipelineStep):
    """
    A processing step that filters the DataFrame to the top k records whose embeddings
    are closest (via cosine similarity) to a given query phrase.
    """
    def __init__(self,
                 query: str,
                 k: int,
                 embedding_function: Callable[[str], List[float]] = openai_embedding_function,
                 embedding_column: str = "embedding"):
        """
        :param query: The query text to compare against.
        :param k: The number of top records to return.
        :param embedding_function: A function that takes text and returns an embedding vector.
                                   This should be the same function used in the GenerateEmbeddingsStep.
        :param embedding_column: The DataFrame column that contains the embeddings.
        """
        self.query = query
        self.k = k
        self.embedding_function = embedding_function
        self.embedding_column = embedding_column

    def cosine_similarity(self, vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Compute the query embedding.
        query_embedding = self.embedding_function(self.query)
        # Calculate cosine similarity for each row.
        df.loc[:, 'similarity'] = df[self.embedding_column].apply(
            lambda emb: self.cosine_similarity(emb, query_embedding)
        )
        # Sort by similarity in descending order and select the top k rows.
        df_sorted = df.sort_values(by='similarity', ascending=False)
        result_df = df_sorted.head(self.k).copy()
        result_df.drop(columns=['similarity'], inplace=True)
        return result_df

##############################################################################
# New Processor: LLMCallWithDataFrame
##############################################################################

class LLMCallWithDataFrame:
    """
    A class that takes an entire DataFrame, constructs a prompt with its content,
    and calls the LLM. It can optionally use a list of fields to include in the prompt.
    """
    def __init__(self, prompt_template: str, fields: List[str] = None):
        self.prompt_template = prompt_template
        self.fields = fields
        self.system_prompt = (
            "You are processing a user request against a set of records.  Please respond to the request as directed, without any additional comments or text.\n\n"
        )
        self.client = OpenAI()

    def create_prompt(self, df: pd.DataFrame) -> str:
        # Generate the record details for each row using the provided fields or all fields.
        def create_record_details(row):
            if self.fields:
                details = {field: row[field] for field in self.fields}
            else:
                details = row.to_dict()
            return "\n".join([f"{key}: {value}" for key, value in details.items()])
        
        record_details = df.apply(create_record_details, axis=1).tolist()
        combined_details = "\n\n".join([f"# new record\n{details}" for details in record_details])
        
        # Generate the full prompt using the combined record details.
        full_prompt = self.prompt_template.format(record_details=combined_details)
        return full_prompt

    def call_llm(self, df: pd.DataFrame) -> str:
        prompt = self.create_prompt(df)
        chat_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        # Return the LLM's response.
        return chat_completion.choices[0].message.content.strip()

##############################################################################
# 4. The Pipeline: Running Steps in Batches
##############################################################################

class DataPipeline:
    """
    A generic pipeline to process data.
    The pipeline runs the data through each of the configured steps.
    """
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.process(df)
        return df
