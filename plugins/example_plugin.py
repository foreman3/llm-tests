"""Example plugin that adds a simple column to the pipeline data."""

from llm_pipeline.llm_methods import PipelineStep
import pandas as pd

class ExamplePluginStep(PipelineStep):
    """Pipeline step that marks the data as processed by the plugin."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``plugin`` column to ``df`` and return the modified frame."""
        df["plugin"] = "loaded"
        return df

