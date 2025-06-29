import unittest
import os
import pandas as pd
from llm_pipeline.llm_methods import (
    LLMCallStep,
    FixedProcessingStep,
    DataPipeline,
    FilterStep,
    GenerateEmbeddingsStep,
    kNNFilterStep,
    LLMCallWithDataFrame,
    AgenticGoalStep,
)
import csv
from typing import List


class TestLLMPipeline(unittest.TestCase):

    def test_classify_pipeline(self):
        # Load test data from CSV file into a DataFrame
        try:
            df = pd.read_csv("./tests/test_data_pipeline.csv", quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError:
            raise
        except FileNotFoundError:
            raise

        # Define a couple of processing steps.

        # Example fixed processing step: add a new column that shows the word count of the description.
        def count_words(df: pd.DataFrame) -> pd.Series:
            return df["description"].apply(lambda x: len(x.split()))

        fixed_step = FixedProcessingStep(count_words, output_key="word_count")

        # Example LLM processing step: use an LLM to extract potential UI change details.
        llm_step = LLMCallStep(
            prompt_template="""Determine if the following development story likely resulted in a UI update or not.
            respond back with only 'yes' or 'no'.

            {record_details}""",
            output_key="ui_change",
            fields=["title", "description", "acceptance_criteria"],
        )

        # Example filter step: filter out rows that are not UI edits.
        def filter_function(df: pd.DataFrame) -> pd.Series:
            return df["ui_change"].str.strip().str.lower() == "yes"

        filter_step = FilterStep(filter_function)

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[fixed_step, llm_step, filter_step])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(df)

        # Verify the results instead of printing
        self.assertEqual(len(processed_df), 3)
        self.assertTrue((processed_df["ui_change"].str.strip().str.lower() == "yes").all())
        self.assertListEqual(processed_df["word_count"].tolist(), [9, 16, 13])

    def test_gen_and_persist_embeddings(self):
        # Load test data from CSV file into a DataFrame
        try:
            df = pd.read_csv("./tests/test_data_pipeline.csv", quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError:
            raise

        # Define a couple of processing steps.

        # Generate one embedding from title & description.
        gen_embed = GenerateEmbeddingsStep(
            output_key="embedding_title_desc", fields=["title", "description"]
        )

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[gen_embed])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(df)
        processed_df.to_pickle("./tests/test_data_pipeline_embeddings.pkl")

        self.assertEqual(len(processed_df), 3)
        self.assertIn("embedding_title_desc", processed_df.columns)
        self.assertTrue(all(len(v) == 32 for v in processed_df["embedding_title_desc"]))
        self.assertTrue(os.path.exists("./tests/test_data_pipeline_embeddings.pkl"))

    def test_reload_embeddings_and_kNN(self):

        # Reload the DataFrame.
        reloaded_df = pd.read_pickle("./tests/test_data_pipeline_embeddings.pkl")

        # Use a kNN filter on one of the embeddings:
        knn_filter = kNNFilterStep(
            query="input form", k=1, embedding_column="embedding_title_desc"
        )

        # Create the pipeline with all steps.
        pipeline = DataPipeline(steps=[knn_filter])

        # Run the pipeline on the DataFrame.
        processed_df = pipeline.run(reloaded_df)

        self.assertEqual(len(processed_df), 1)
        self.assertEqual(processed_df.iloc[0]["id"], "B-29990")

        llm_eval = LLMCallWithDataFrame(
            prompt_template="""What UIs changes are being implemented?'.

            {record_details}""",
            fields=["title", "description", "acceptance_criteria"],
        )
        self.assertEqual(llm_eval.call_llm(processed_df), "LLM unavailable")


class TestAgenticStep(unittest.TestCase):
    def test_agentic_goal_step(self):
        df = pd.DataFrame({"title": ["alpha", "beta"]})

        def count_chars(context, _):
            """Return character count of the title"""
            return len(context["title"])

        agent = AgenticGoalStep(
            goal="count characters in title",
            per_row=True,
            tools={"count_chars": count_chars},
        )

        pipeline = DataPipeline([agent])
        result = pipeline.run(df)
        assert result["agent_output"].tolist() == [5, 4]

    def test_agentic_multi_tool(self):
        df = pd.DataFrame({"num": [3]})

        def double(ctx, _):
            return ctx["num"] * 2

        def minus_one(val, _):
            return val - 1

        agent = AgenticGoalStep(
            goal="double then minus one",
            per_row=True,
            tools={"double": double, "minus_one": minus_one},
        )

        pipeline = DataPipeline([agent])
        result = pipeline.run(df)
        assert result["agent_output"].tolist() == [5]

    def test_agentic_with_mcp_server(self):
        df = pd.DataFrame({"title": ["hello"]})

        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading, json

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/tools":
                    tools = {"echo": "echo back"}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(tools).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                context = data.get("context", {})
                if self.path == "/echo":
                    result = {**context, "server": "called"}
                else:
                    result = {"error": "unknown"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

        server = HTTPServer(("localhost", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        url = f"http://{server.server_address[0]}:{server.server_address[1]}"

        agent = AgenticGoalStep(goal="echo", per_row=True, mcp_servers=[url])

        pipeline = DataPipeline([agent])
        result = pipeline.run(df)
        server.shutdown()
        thread.join()
        assert result["agent_output"].iloc[0]["server"] == "called"


if __name__ == "__main__":
    unittest.main()
