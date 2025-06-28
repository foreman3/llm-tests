import pandas as pd
from llm_pipeline.pipeline import GenerateEmbeddingsStep, DataPipeline
import csv

# Load test data from CSV file into a DataFrame
try:
    df = pd.read_csv('./data/test_data_pipeline.csv', quoting=csv.QUOTE_ALL)
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
    raise

# Define a couple of processing steps.

# Generate one embedding from title & description.
gen_embed = GenerateEmbeddingsStep(
    output_key="embedding_title_desc",
    fields=["title", "description"]
)

# Create the pipeline with all steps.
pipeline = DataPipeline(steps=[gen_embed])

# Run the pipeline on the DataFrame.
processed_df = pipeline.run(df)
count = len(processed_df)

print(f"Processed record count: {count}")
processed_df.to_pickle('./data/monsters_with_embeddings.pkl')
