from llm_pipeline.pipeline import PipelineStep
import pandas as pd

class ExamplePluginStep(PipelineStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['plugin'] = 'loaded'
        return df
