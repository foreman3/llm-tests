import pandas as pd
from llm_pipeline.pipeline import DataPipeline, kNNFilterStep, LLMCallWithDataFrame
import csv


# Reload the DataFrame.
reloaded_df = pd.read_pickle('./data/monsters_with_embeddings.pkl')

# Use a kNN filter on one of the embeddings:
knn_filter = kNNFilterStep(
    query="input form",
    k=3,
    embedding_column="embedding_title_desc"
)

# Create the pipeline with all steps.
pipeline = DataPipeline(steps=[knn_filter])

# Run the pipeline on the DataFrame.
processed_df = pipeline.run(reloaded_df)

print(processed_df[['id', 'title']].to_string())

llm_eval = LLMCallWithDataFrame(
    prompt_template="""What UI updates are being preformed?  Ignore records that not UI updates'.

    {record_details}""",
    fields=["title", "description", "acceptance_criteria"],
)
print(llm_eval.call_llm(processed_df))