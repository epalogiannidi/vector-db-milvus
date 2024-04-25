from milvus_handler import MilvusHandler
from utils import load_config, set_up_logger

CONFIG_PATH = "config.yaml"

if __name__ == "__main__":
    logger = set_up_logger()

    logger.info("Loading configuration....")
    config = load_config(CONFIG_PATH)

    logger.info("Instantiating milvus handler...")
    mh = MilvusHandler(config, logger)

    logger.info("Connecting to milvus...")
    mh.connect()

    logger.info("Checking whether the collection already exists...")
    mh.check_collection_existence()

    logger.info("Defining collection schema...")
    mh.define_schema()

    logger.info(f"Connecting to collection: {mh.collection_name}")
    mh.connect_to_collection()

    if mh.milvus_collection is not None:
        if mh.milvus_collection.num_entities == 0:
            logger.info("Inserting the first entities to the collection...")
            with open(config["data"], "r") as file:
                lines = file.readlines()

            insert_result = mh.insert_data(lines, mh.milvus_collection.num_entities)

    logger.info("Loading the collection into memory...")
    mh.milvus_collection.load()

    if mh.milvus_collection is not None:
        logger.info("Search for similar texts based on the embeddings....")
        hits = mh.search(
            text="The quick brown fox jumps over the lazy dog.",
            limit=3,
            output_fields=["pk", "sentence"],
        )

    if mh.milvus_collection is not None:
        logger.info("Insert an extra entity")
        insert_result2 = mh.insert_data(
            ["This is a new entry"], mh.milvus_collection.num_entities
        )

    logger.info("Dropping the collection")
    mh.drop_db()
