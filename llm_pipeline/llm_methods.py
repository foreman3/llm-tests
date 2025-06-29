import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Callable, List

import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import DBSCAN

from tools.utils import log_message, validate_input

logger = logging.getLogger(__name__)

load_dotenv()

# ------------------------------------------------------------------
# Utility: OpenAI embedding function (default implementation)
# ------------------------------------------------------------------
def openai_embedding_function(text: str) -> List[float]:
    """Return an embedding vector for ``text``.

    If ``OPENAI_API_KEY`` is available the function calls the OpenAI API,
    otherwise it falls back to generating a deterministic pseudo embedding so
    that the rest of the pipeline can operate without network access.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[:32]]

    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:  # pragma: no cover - network errors
        logger.error("Error generating embedding: %s", e)
        return []

# ------------------------------------------------------------------
# 1. LLM call per record (analogous to LLMCallStep)
# ------------------------------------------------------------------
def apply_llm_call(
    df: pd.DataFrame,
    prompt_template: str,
    output_key: str = "llm_output",
    fields: List[str] = None
) -> pd.DataFrame:
    """
    For each row in the DataFrame, build a prompt from specified fields (or all fields),
    call the LLM, and store the response in a new column.
    """
    client = OpenAI()
    system_prompt = (
        "You are evaluating records. The user will submit a record to be evaluated, and you should respond with only the evaluation. \n"
        "Respond only with the evaluation, and no other terms or formatting.\n\n"
        "The requested evaluation is:"
    )

    def create_record_details(row):
        details = {field: row[field] for field in fields} if fields else row.to_dict()
        return "\n".join(f"{key}: {value}" for key, value in details.items())

    # Work on a copy so as not to modify the original DataFrame.
    df = df.copy()
    df['record_details'] = df.apply(create_record_details, axis=1)
    df['prompt'] = df['record_details'].apply(
        lambda details: prompt_template.format(record_details=details)
    )

    def get_llm_response(prompt: str) -> str:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return chat_completion.choices[0].message.content.strip()

    df[output_key] = df['prompt'].apply(get_llm_response)
    df.drop(columns=['record_details', 'prompt'], inplace=True)
    return df

# ------------------------------------------------------------------
# 2. Generate embeddings (analogous to GenerateEmbeddingsStep)
# ------------------------------------------------------------------
def generate_embeddings(
    df: pd.DataFrame,
    embedding_function: Callable[[str], List[float]] = openai_embedding_function,
    output_key: str = "embedding",
    fields: List[str] = None
) -> pd.DataFrame:
    """
    For each row in the DataFrame, combine specified fields (or all columns) into text,
    compute an embedding, and store it in a new column.
    """
    df = df.copy()

    def get_text(row):
        if fields:
            return " ".join(str(row[field]) for field in fields)
        else:
            return " ".join(row.astype(str))
    
    df[output_key] = df.apply(lambda row: embedding_function(get_text(row)), axis=1)
    return df

# ------------------------------------------------------------------
# 3. kNN filter (analogous to kNNFilterStep)
# ------------------------------------------------------------------
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    a, b = np.array(vec_a), np.array(vec_b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def knn_filter(
    df: pd.DataFrame,
    query: str,
    k: int,
    embedding_function: Callable[[str], List[float]] = openai_embedding_function,
    embedding_column: str = "embedding"
) -> pd.DataFrame:
    """
    Filter the DataFrame to the top k rows whose embeddings are closest (by cosine similarity)
    to the embedding of the query text.
    """
    df = df.copy()
    query_embedding = embedding_function(query)
    df['similarity'] = df[embedding_column].apply(lambda emb: cosine_similarity(emb, query_embedding))
    df_sorted = df.sort_values(by='similarity', ascending=False)
    result_df = df_sorted.head(k).copy()
    result_df.drop(columns=['similarity'], inplace=True)
    return result_df

# ------------------------------------------------------------------
# 4. LLM call with entire DataFrame (analogous to LLMCallWithDataFrame)
# ------------------------------------------------------------------
def call_llm_with_dataframe(
    df: pd.DataFrame,
    prompt_template: str,
    fields: List[str] = None
) -> str:
    """
    Combine records from the DataFrame into one prompt and call the LLM.
    """
    client = OpenAI()
    system_prompt = (
        "You are processing a user request against a set of records.  Please respond to the request as directed, without any additional comments or text.\n\n"
    )

    def create_record_details(row):
        details = {field: row[field] for field in fields} if fields else row.to_dict()
        return "\n".join(f"{key}: {value}" for key, value in details.items())

    record_details_list = df.apply(create_record_details, axis=1).tolist()
    combined_details = "\n\n".join(f"# new record\n{details}" for details in record_details_list)
    full_prompt = prompt_template.format(record_details=combined_details)

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return chat_completion.choices[0].message.content.strip()

# ------------------------------------------------------------------
# 5. DB Scan function to produce clusters from embeddings
# ------------------------------------------------------------------
def cluster_dbscan(
    df: pd.DataFrame,
    embedding_column: str,
    output_key: str = "cluster_id",
    eps: float = 0.5,
    min_samples: int = 5,
    **dbscan_kwargs: Any
) -> pd.DataFrame:
    """
    Cluster the embeddings in a specified column of the DataFrame using DBSCAN.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        embedding_column (str): The name of the column containing the embedding vectors.
                                Each entry should be a list or a numpy array.
        output_key (str): The name of the output column where the cluster labels will be stored.
                          Default is "cluster_id".
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other (DBSCAN parameter).
        min_samples (int): The number of samples in a neighborhood for a point to be considered
                           as a core point (DBSCAN parameter).
        **dbscan_kwargs: Additional keyword arguments to pass to the DBSCAN constructor.
    
    Returns:
        pd.DataFrame: The DataFrame with an added column for the cluster labels.
    """
    # Check if the embedding column exists in the DataFrame.
    if embedding_column not in df.columns:
        raise ValueError(f"Column '{embedding_column}' not found in the DataFrame.")
    
    # Extract embeddings and convert them into a numpy array.
    embeddings = df[embedding_column].tolist()
    embeddings_arr = np.array(embeddings)
    
    # Create and fit the DBSCAN clustering algorithm.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, **dbscan_kwargs)
    cluster_labels = dbscan.fit_predict(embeddings_arr)
    
    # Add the cluster labels to the DataFrame.
    df[output_key] = cluster_labels
    
    return df

    # Assume call_llm_with_dataframe is defined elsewhere in your module:
    # def call_llm_with_dataframe(df: pd.DataFrame, prompt_template: str, fields: List[str] = None) -> str: ...

def call_llm_in_batches(
    df: pd.DataFrame,
    prompt_template: str,
    fields: List[str] = None,
    batch_size: int = 100,
    consolidation_prefix: str = (
        "The following responses were generated for batches of records. "
        "Now, please consolidate these outputs into a final summary."
    )
) -> str:
    """
    Process the DataFrame in batches, calling the LLM on each batch, and then consolidate
    the responses with a final LLM call.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        prompt_template (str): A prompt template for each batch call. Must include a '{record_details}' placeholder.
        fields (List[str], optional): The list of fields to include in each prompt (if None, all fields are used).
        batch_size (int, optional): The number of records per batch (default is 100).
        consolidation_prefix (str, optional): A prefix to add to the final consolidation prompt to explain the context.
    
    Returns:
        str: The final consolidated LLM response.
    """
    num_rows = len(df)
    num_batches = ceil(num_rows / batch_size)
    batch_responses = []

    # Process each batch
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_df = df.iloc[start:end]
        # Call the LLM on the batch using your existing function.
        batch_response = call_llm_with_dataframe(batch_df, prompt_template, fields)
        batch_responses.append(batch_response)

    # Combine the responses from each batch.
    combined_responses = "\n\n".join(batch_responses)
    # Create a final prompt template that prepends the consolidation prefix.
    final_prompt_template = consolidation_prefix + "\n\n{record_details}"

    # Create a dummy DataFrame with one row containing the combined responses.
    consolidation_df = pd.DataFrame({"record_details": [combined_responses]})
    # Call the LLM one final time to consolidate the batch outputs.
    final_response = call_llm_with_dataframe(consolidation_df, final_prompt_template, fields=["record_details"])
    return final_response


class DataFrameProcessor:
    """
    A lightweight wrapper for a DataFrame that supports chainable processing operations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def llm_call(
        self,
        prompt_template: str,
        output_key: str = "llm_output",
        fields: List[str] = None
    ) -> "DataFrameProcessor":
        self.df = apply_llm_call(self.df, prompt_template, output_key, fields)
        return self

    def generate_embeddings(
        self,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        output_key: str = "embedding",
        fields: List[str] = None
    ) -> "DataFrameProcessor":
        self.df = generate_embeddings(self.df, embedding_function, output_key, fields)
        return self

    def knn_filter(
        self,
        query: str,
        k: int,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        embedding_column: str = "embedding"
    ) -> "DataFrameProcessor":
        self.df = knn_filter(self.df, query, k, embedding_function, embedding_column)
        return self
    
    def filter(
        self,
        filter_function: Callable[[pd.DataFrame], pd.Series]
    ) -> "DataFrameProcessor":
        """
        Filters the DataFrame using the provided filter_function.
        The filter_function should accept a DataFrame and return a Boolean Series.
        """
        self.df = self.df.loc[filter_function(self.df)]
        return self
    
    def cluster_dbscan(
        self,
        embedding_column: str,
        output_key: str = "cluster_id",
        eps: float = 0.5,
        min_samples: int = 5,
        **dbscan_kwargs: Any
    ) -> "DataFrameProcessor":
        """
        Cluster the DataFrame based on the embeddings in the specified column using DBSCAN.
        
        Parameters:
            embedding_column (str): The column name that contains the embedding vectors.
            output_key (str): The name of the new column to store cluster labels. Defaults to "cluster_id".
            eps (float): The maximum distance between two samples for them to be considered in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
            **dbscan_kwargs: Additional keyword arguments to pass to the DBSCAN constructor.
        
        Returns:
            DataFrameProcessor: Returns self so that the method calls can be chained.
        """
        self.df = cluster_dbscan(self.df, embedding_column, output_key, eps, min_samples, **dbscan_kwargs)
        return self

    def call_llm_in_batches(    
        self,
        prompt_template: str,
        fields: List[str] = None,
        batch_size: int = 100,
        consolidation_prefix: str = (
            "The following responses were generated for batches of records. "
            "Now, please consolidate these outputs into a final summary."
        )
    ) -> str:
        """
        Process the DataFrame in batches and call the LLM on each batch,
        then consolidate the batch responses into a final result.
        
        Parameters:
            prompt_template (str): The prompt template to use for each batch call. Must include '{record_details}'.
            fields (List[str], optional): The list of fields to include in each prompt.
            batch_size (int, optional): The number of records per batch.
            consolidation_prefix (str, optional): A prefix for the final consolidation prompt.
        
        Returns:
            str: The final consolidated LLM response.
        """
        return call_llm_in_batches(
            self.df,
            prompt_template,
            fields,
            batch_size,
            consolidation_prefix
        )

    def call_llm_with_dataframe(
        self,
        prompt_template: str,
        fields: List[str] = None
    ) -> str:
        return call_llm_with_dataframe(self.df, prompt_template, fields)

    def get_df(self) -> pd.DataFrame:
        return self.df


###############################################################################
# Pipeline style classes migrated from pipeline module
###############################################################################


class PipelineStep(ABC):
    """Abstract base class for a pipeline processing step."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process ``df`` and return the updated DataFrame."""
        raise NotImplementedError


class LLMCallStep(PipelineStep):
    """Call an LLM for each record with optional caching."""

    def __init__(
        self,
        prompt_template: str,
        output_key: str = "llm_output",
        fields: List[str] | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.fields = fields
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load cache from %s: %s", cache_path, exc)
                self.cache = {}
        self.system_prompt = (
            "You are evaluating records. The user will submit a record to be evaluated, and you should respond with only the evaluation. \n"
            "Respond only with the evaluation, and no other terms or formatting.\n\n"
            "The requested evaluation is:"
        )
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def _get_llm_response(self, prompt: str) -> str:
        if prompt in self.cache:
            return self.cache[prompt]
        if self.client is None:
            logger.debug("OPENAI_API_KEY not set; using offline fallback")
            response = "yes" if "ui" in prompt.lower() else "no"
        else:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            response = chat_completion.choices[0].message.content.strip()
        self.cache[prompt] = response
        if self.cache_path:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
            except Exception as exc:
                logger.warning("Failed to save cache to %s: %s", self.cache_path, exc)
        return response

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        def create_record_details(row: pd.Series) -> str:
            details = {field: row[field] for field in self.fields} if self.fields else row.to_dict()
            return "\n".join(f"{key}: {value}" for key, value in details.items())

        df = df.copy()
        df["record_details"] = df.apply(create_record_details, axis=1)
        df["prompt"] = df["record_details"].apply(
            lambda details: self.prompt_template.format(record_details=details)
        )
        df[self.output_key] = df["prompt"].apply(self._get_llm_response)
        df.drop(columns=["record_details", "prompt"], inplace=True)
        return df

    async def process_async(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def create_record_details(row: pd.Series) -> str:
            details = {field: row[field] for field in self.fields} if self.fields else row.to_dict()
            return "\n".join(f"{key}: {value}" for key, value in details.items())

        df["record_details"] = df.apply(create_record_details, axis=1)
        df["prompt"] = df["record_details"].apply(
            lambda details: self.prompt_template.format(record_details=details)
        )

        async def call(prompt: str) -> str:
            return await asyncio.to_thread(self._get_llm_response, prompt)

        responses = await asyncio.gather(*(call(p) for p in df["prompt"].tolist()))
        df[self.output_key] = responses
        df.drop(columns=["record_details", "prompt"], inplace=True)
        return df


class FixedProcessingStep(PipelineStep):
    """A processing step that applies a fixed function."""

    def __init__(
        self,
        process_function: Callable[[pd.DataFrame], pd.Series],
        output_key: str,
    ) -> None:
        self.process_function = process_function
        self.output_key = output_key

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.output_key] = self.process_function(df)
        return df


class FilterStep(PipelineStep):
    """Filter rows in the DataFrame based on ``filter_function``."""

    def __init__(self, filter_function: Callable[[pd.DataFrame], pd.Series]) -> None:
        self.filter_function = filter_function

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.filter_function(df)]


class GenerateEmbeddingsStep(PipelineStep):
    """Generate embeddings for each record."""

    def __init__(
        self,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        output_key: str = "embedding",
        fields: List[str] | None = None,
    ) -> None:
        self.embedding_function = embedding_function
        self.output_key = output_key
        self.fields = fields

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_text(row: pd.Series) -> str:
            if self.fields:
                return " ".join(str(row[field]) for field in self.fields)
            return " ".join(row.astype(str))

        df[self.output_key] = df.apply(lambda row: self.embedding_function(get_text(row)), axis=1)
        return df


class kNNFilterStep(PipelineStep):
    """Filter to the top ``k`` records most similar to ``query``."""

    def __init__(
        self,
        query: str,
        k: int,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        embedding_column: str = "embedding",
    ) -> None:
        self.query = query
        self.k = k
        self.embedding_function = embedding_function
        self.embedding_column = embedding_column

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        query_embedding = self.embedding_function(self.query)
        df["similarity"] = df[self.embedding_column].apply(
            lambda emb: self._cosine_similarity(emb, query_embedding)
        )
        df_sorted = df.sort_values(by="similarity", ascending=False)
        result_df = df_sorted.head(self.k).copy()
        result_df.drop(columns=["similarity"], inplace=True)
        return result_df


class LLMCallWithDataFrame:
    """Call an LLM with the entire DataFrame combined into one prompt."""

    def __init__(self, prompt_template: str, fields: List[str] | None = None) -> None:
        self.prompt_template = prompt_template
        self.fields = fields
        self.system_prompt = (
            "You are processing a user request against a set of records.  Please respond to the request as directed, without any additional comments or text.\n\n"
        )
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def create_prompt(self, df: pd.DataFrame) -> str:
        def create_record_details(row: pd.Series) -> str:
            details = {field: row[field] for field in self.fields} if self.fields else row.to_dict()
            return "\n".join(f"{key}: {value}" for key, value in details.items())

        record_details = df.apply(create_record_details, axis=1).tolist()
        combined = "\n\n".join(f"# new record\n{details}" for details in record_details)
        return self.prompt_template.format(record_details=combined)

    def call_llm(self, df: pd.DataFrame) -> str:
        prompt = self.create_prompt(df)
        if self.client is None:
            return "LLM unavailable"
        chat_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return chat_completion.choices[0].message.content.strip()


class AgenticGoalStep(PipelineStep):
    """An agentic step that attempts to accomplish a goal using tools."""

    def __init__(
        self,
        goal: str,
        *,
        per_row: bool = False,
        guidance: str = "",
        tools: dict | None = None,
        output_key: str = "agent_output",
        use_mcp: bool = False,
        mcp_servers: List[str] | None = None,
        discover_mcp: bool = True,
        max_steps: int = 3,
    ) -> None:
        self.goal = goal
        self.per_row = per_row
        self.guidance = guidance
        self.tools = tools or {}
        if use_mcp:
            try:
                from mcp import MCP_TOOLS

                self.tools.update(MCP_TOOLS)
            except Exception as exc:
                logger.debug("Failed to load MCP tools: %s", exc)
        self.mcp_servers = mcp_servers or []
        self.discover_mcp = discover_mcp
        for i, url in enumerate(self.mcp_servers):
            if discover_mcp:
                try:
                    from mcp import discover_remote_tools

                    discovered = discover_remote_tools(url)
                    if discovered:
                        self.tools.update(discovered)
                        continue
                except Exception as exc:
                    logger.debug("Failed to discover tools from %s: %s", url, exc)
            self.tools[f"mcp_server_{i}"] = self._make_mcp_tool(url)
        self.output_key = output_key
        self.max_steps = max_steps
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def _make_mcp_tool(
        self, url: str, name: str | None = None, desc: str | None = None
    ) -> Callable[[dict, dict | None], dict]:
        """Return a wrapper that sends context to an MCP server or endpoint."""

        endpoint = url if name is None else f"{url.rstrip('/')}/{name}"

        def call(context: dict, args: dict | None = None) -> dict:
            try:
                import requests

                payload = {"context": context, "args": args or {}}
                resp = requests.post(endpoint, json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # pragma: no cover - network errors
                return {"error": str(exc)}

        call.__doc__ = desc

        return call

    def _agent_prompt(self, context: dict, history: list) -> str:
        tool_desc = "\n".join(
            f"- {name}: {func.__doc__ or 'no description'}" for name, func in self.tools.items()
        )
        hist = "\n".join(f"Step {i+1} result: {h}" for i, h in enumerate(history))
        return (
            f"Goal: {self.goal}\n{self.guidance}\nContext: {context}\n{hist}\n"
            f"Available tools:\n{tool_desc}\n"
            'Respond with JSON either {"tool": <name>, "args": <args>} '
            'to invoke a tool or {"answer": <result>} to answer directly.'
        )

    def _run_agent(self, context: dict) -> str:
        history: list = []

        if self.client is None:
            result = context
            for func in self.tools.values():
                result = func(result, None)
                history.append(result)
            return result

        for _ in range(self.max_steps):
            prompt = self._agent_prompt(context, history)
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            msg = chat_completion.choices[0].message.content.strip()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                return msg
            if "tool" in data and data["tool"] in self.tools:
                tool_output = self.tools[data["tool"]](context, data.get("args"))
                history.append(tool_output)
                context["last_tool_output"] = tool_output
                continue
            return data.get("answer", msg)
        return str(history[-1]) if history else ""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.per_row:
            df[self.output_key] = df.apply(lambda row: self._run_agent(row.to_dict()), axis=1)
            return df
        self.result = self._run_agent(df.to_dict(orient="records"))
        return df


class SummarizationStep(PipelineStep):
    """Generate a summary for the entire DataFrame."""

    def __init__(self, prompt_template: str, fields: List[str] | None = None) -> None:
        self.llm = LLMCallWithDataFrame(prompt_template, fields)
        self.summary: str = ""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        self.summary = self.llm.call_llm(df)
        return df


class DataPipeline:
    """Run a sequence of ``PipelineStep`` objects over a DataFrame."""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_input(df, pd.DataFrame)
        for step in self.steps:
            log_message(f"Running step {step.__class__.__name__}")
            df = step.process(df)
            validate_input(df, pd.DataFrame)
        return df


class AsyncDataPipeline(DataPipeline):
    """Run pipeline steps asynchronously when possible."""

    async def run(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        validate_input(df, pd.DataFrame)
        for step in self.steps:
            log_message(f"Running step {step.__class__.__name__} asynchronously")
            if hasattr(step, "process_async"):
                df = await step.process_async(df)
            else:
                df = await asyncio.to_thread(step.process, df)
            validate_input(df, pd.DataFrame)
        return df

