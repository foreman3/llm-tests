import pandas as pd
from llm_pipeline.llm_methods import GenerateEmbeddingsStep, DataPipeline
import csv
import logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Generate embeddings for the monster dataset."""

    # Load test data from CSV file into a DataFrame
    try:
        df = pd.read_csv('./data/test_data_pipeline.csv', quoting=csv.QUOTE_ALL)
    except pd.errors.ParserError as e:
        logger.error("Error reading CSV file: %s", e)
        raise

    # Generate one embedding from title & description.
    gen_embed = GenerateEmbeddingsStep(
        output_key="embedding_title_desc",
        fields=["title", "description"]
    )

    pipeline = DataPipeline(steps=[gen_embed])

    processed_df = pipeline.run(df)
    count = len(processed_df)

    logger.info("Processed record count: %s", count)
    processed_df.to_pickle('./data/monsters_with_embeddings.pkl')


if __name__ == "__main__":
    main()
