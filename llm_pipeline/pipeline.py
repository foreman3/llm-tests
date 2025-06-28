import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, List
from openai import OpenAI
from dotenv import load_dotenv
import os
import hashlib
import numpy as np
import json
import asyncio
from tools.utils import log_message, validate_input

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
    """Call an LLM for each record with optional caching."""

    def __init__(
        self,
        prompt_template: str,
        output_key: str = "llm_output",
        fields: List[str] = None,
        cache_path: str | None = None,
    ):
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.fields = fields
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        self.system_prompt = (
            "You are evaluating records. The user will submit a record to be evaluated, and you should respond with only the evaluation. \n"
            "Respond only with the evaluation, and no other terms or formatting.\n\n"
            "The requested evaluation is:"
        )
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Generate the record details for each row using the provided fields or all fields.
        def create_record_details(row):
            if self.fields:
                details = {field: row[field] for field in self.fields}
            else:
                details = row.to_dict()
            return "\n".join([f"{key}: {value}" for key, value in details.items()])

        df.loc[:, "record_details"] = df.apply(create_record_details, axis=1)

        # Generate the prompt for each row using the record details.
        df.loc[:, "prompt"] = df["record_details"].apply(
            lambda details: self.prompt_template.format(record_details=details)
        )

        # Define a helper function to call the LLM for each prompt with caching.
        def get_llm_response(prompt: str) -> str:
            if prompt in self.cache:
                return self.cache[prompt]
            if self.client is None:
                # Simple offline heuristic
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
                except Exception:
                    pass
            return response

        # Use the helper function with .apply() to get the output.
        df.loc[:, self.output_key] = df["prompt"].apply(get_llm_response)

        # Optionally, drop the temporary columns.
        df.drop(columns=["record_details", "prompt"], inplace=True)

        return df

    async def process_async(self, df: pd.DataFrame) -> pd.DataFrame:
        def create_record_details(row):
            if self.fields:
                details = {field: row[field] for field in self.fields}
            else:
                details = row.to_dict()
            return "\n".join([f"{key}: {value}" for key, value in details.items()])

        df = df.copy()
        df.loc[:, "record_details"] = df.apply(create_record_details, axis=1)
        df.loc[:, "prompt"] = df["record_details"].apply(
            lambda details: self.prompt_template.format(record_details=details)
        )

        async def call(prompt: str) -> str:
            return await asyncio.to_thread(get_llm_response, prompt)

        def get_llm_response(prompt: str) -> str:
            if prompt in self.cache:
                return self.cache[prompt]
            if self.client is None:
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
                except Exception:
                    pass
            return response

        responses = await asyncio.gather(*(call(p) for p in df["prompt"].tolist()))
        df.loc[:, self.output_key] = responses
        df.drop(columns=["record_details", "prompt"], inplace=True)
        return df


class FixedProcessingStep(PipelineStep):
    """
    A processing step that uses a fixed function to process the DataFrame.
    """

    def __init__(
        self,
        process_function: Callable[[pd.DataFrame], pd.Series],
        output_key: str,
    ):
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
    """Return an embedding vector for ``text``.

    If ``OPENAI_API_KEY`` is available the function calls the OpenAI API,
    otherwise it falls back to generating a deterministic pseudo embedding so
    that the rest of the pipeline can operate without network access.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Deterministic embedding based on a hash of the text
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Convert bytes to floats in range [0, 1]
        return [b / 255 for b in digest[:32]]

    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
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

    def __init__(
        self,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        output_key: str = "embedding",
        fields: List[str] = None,
    ):
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
        df.loc[:, self.output_key] = df.apply(
            lambda row: self.embedding_function(get_text(row)), axis=1
        )
        return df


##############################################################################
# New Processor: kNNFilterStep
##############################################################################


class kNNFilterStep(PipelineStep):
    """
    A processing step that filters the DataFrame to the top k records whose embeddings
    are closest (via cosine similarity) to a given query phrase.
    """

    def __init__(
        self,
        query: str,
        k: int,
        embedding_function: Callable[[str], List[float]] = openai_embedding_function,
        embedding_column: str = "embedding",
    ):
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
        df.loc[:, "similarity"] = df[self.embedding_column].apply(
            lambda emb: self.cosine_similarity(emb, query_embedding)
        )
        # Sort by similarity in descending order and select the top k rows.
        df_sorted = df.sort_values(by="similarity", ascending=False)
        result_df = df_sorted.head(self.k).copy()
        result_df.drop(columns=["similarity"], inplace=True)
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
        self.system_prompt = "You are processing a user request against a set of records.  Please respond to the request as directed, without any additional comments or text.\n\n"
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def create_prompt(self, df: pd.DataFrame) -> str:
        # Generate the record details for each row using the provided fields or all fields.
        def create_record_details(row):
            if self.fields:
                details = {field: row[field] for field in self.fields}
            else:
                details = row.to_dict()
            return "\n".join([f"{key}: {value}" for key, value in details.items()])

        record_details = df.apply(create_record_details, axis=1).tolist()
        combined_details = "\n\n".join(
            [f"# new record\n{details}" for details in record_details]
        )

        # Generate the full prompt using the combined record details.
        full_prompt = self.prompt_template.format(record_details=combined_details)
        return full_prompt

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


##############################################################################
# Agentic Goal Step
##############################################################################


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
        max_steps: int = 3,
    ):
        self.goal = goal
        self.per_row = per_row
        self.guidance = guidance
        self.tools = tools or {}
        if use_mcp:
            try:
                from mcp import MCP_TOOLS

                self.tools.update(MCP_TOOLS)
            except Exception:
                pass
        self.mcp_servers = mcp_servers or []
        for i, url in enumerate(self.mcp_servers):
            self.tools[f"mcp_server_{i}"] = self._make_mcp_tool(url)
        self.output_key = output_key
        self.max_steps = max_steps
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def _make_mcp_tool(self, url: str) -> Callable[[dict, dict | None], dict]:
        """Return a wrapper that sends context to an MCP server."""

        def call(context: dict, args: dict | None = None) -> dict:
            try:
                import requests

                payload = {"context": context, "args": args or {}}
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                return {"error": str(exc)}

        return call

    def _agent_prompt(self, context: dict, history: list) -> str:
        tool_desc = "\n".join(
            f"- {name}: {func.__doc__ or 'no description'}"
            for name, func in self.tools.items()
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
            df.loc[:, self.output_key] = df.apply(
                lambda row: self._run_agent(row.to_dict()), axis=1
            )
            return df
        self.result = self._run_agent(df.to_dict(orient="records"))
        return df


##############################################################################
# Summarization/Report Generation Step
##############################################################################


class SummarizationStep(PipelineStep):
    """Generate a summary for the entire DataFrame."""

    def __init__(self, prompt_template: str, fields: List[str] | None = None):
        self.llm = LLMCallWithDataFrame(prompt_template, fields)
        self.summary: str = ""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        self.summary = self.llm.call_llm(df)
        return df


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
