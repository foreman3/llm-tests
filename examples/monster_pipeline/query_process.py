import pandas as pd
from llm_pipeline.llm_methods import DataPipeline, kNNFilterStep, LLMCallWithDataFrame
import csv
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Query the vector store and evaluate results."""

    reloaded_df = pd.read_pickle('./data/monsters_with_embeddings.pkl')

    knn_filter = kNNFilterStep(
        query="input form",
        k=3,
        embedding_column="embedding_title_desc",
    )

    pipeline = DataPipeline(steps=[knn_filter])
    processed_df = pipeline.run(reloaded_df)

    logger.info("%s", processed_df[['id', 'title']].to_string())

    llm_eval = LLMCallWithDataFrame(
        prompt_template="""What UI updates are being preformed?  Ignore records that not UI updates'.

    {record_details}""",
        fields=["title", "description", "acceptance_criteria"],
    )
    logger.info("%s", llm_eval.call_llm(processed_df))


if __name__ == "__main__":
    main()
