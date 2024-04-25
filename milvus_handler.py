import time
from logging import RootLogger
from typing import Dict, List

import pymilvus.orm.mutation
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer


class MilvusHandler:
    """
    Handles the Milvus database. Implements a set of functions for interacting with the
    collections that are stored on the milvus db.

    Attributes
    ----------
    config: Dict
        The configuration object. It has been loaded by a yaml file
    host: str
        The milvus host
    port: str
        The milvus port
    alias: str
        The milvus alias
    description: str
        The collection description
    collection_name: str
        The collection name
    collection_exists: bool
        Whether the collection exists in the milvus db or not
    schema: CollectionSchema
        The schema of the collection. It defines the fields and their type
    milvus_collection: Collection
        The milvus collection
    model: SentenceTransformer
        A sentence embeddings model that is used to convert text to embeddings
    logger: RootLogger
        The logger used to keep track of the logs
    """

    def __init__(self, config: Dict, logger: RootLogger):
        self.config = config
        self.host = config["milvus"]["host"]
        self.port = config["milvus"]["port"]
        self.alias = config["milvus"]["alias"]
        self.description = config["milvus"]["description"]
        self.collection_name = config["milvus"]["collection_name"]
        self.collection_exists = False
        self.schema = None
        self.milvus_collection: Collection | None = None

        self.model = SentenceTransformer(config["model"]["name"])
        self.logger = logger

    def check_collection_existence(self) -> None:
        """
        Assigns a boolean to the appropriate class attribute based on
        whether the collection exists or not.
        """
        self.collection_exists = utility.has_collection(self.collection_name)

    def define_schema(self) -> None:
        """
        Defines the collection schema based on the configuration values.
        Assumptions:
            - The primary key is named 'pk'
            - If a field named 'embeddings' exists then the dimension is retrieved by the embeddings model
            - Each field in the configuration file should have a name, a dtype and a max_length
        """
        fields = []

        for field in self.config["milvus"]["fields"]:
            match field["name"]:
                case "pk":
                    fields.append(
                        FieldSchema(
                            name=field["name"],
                            dtype=DataType[field["dtype"]],
                            max_length=field["max_length"],
                            is_primary=True,
                            auto_id=False,
                        )
                    )
                case "embeddings":
                    fields.append(
                        FieldSchema(
                            name=field["name"],
                            dtype=DataType[field["dtype"]],
                            dim=self.model.get_sentence_embedding_dimension(),
                        )
                    )
                case _:
                    fields.append(
                        FieldSchema(
                            name=field["name"],
                            dtype=DataType[field["dtype"]],
                            max_length=field["max_length"],
                        )
                    )
        self.schema = CollectionSchema(fields, self.description)

    def connect_to_collection(self) -> None:
        """
        If the collection exists, the handler connects to it.
        Otherwise, a new collection is being created, along with an index for the embeddings field.
        """
        if self.collection_exists:
            self.logger.info(f"Collection: {self.collection_name} already exists.")
            milvus_collection = Collection(self.collection_name)
        else:
            milvus_collection = Collection(
                self.collection_name, self.schema, consistency_level="Strong"
            )

            # Create an index
            self.logger.info("Start Creating index IVF_FLAT")
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }
            milvus_collection.create_index("embeddings", index)

        self.milvus_collection = milvus_collection

    def insert_data(
        self, raw_data: List[str], starting_idx: int
    ) -> pymilvus.orm.mutation.MutationResult:
        """
        Receives a list with raw texts, generates the sentence embeddings and creates the entities
        that will be inserted to the collection, based on the collection schema.

        :param raw_data: List[str]
            The list with the raw sentences
        :param starting_idx: int
            The starting index that will be used to set values to the primary keys
        :return: MutationResult
            The pymilvus response after inserting the data to the collection

        """
        s_embeddings = [self.model.encode(sentence.strip()) for sentence in raw_data]

        entities = [
            [str(i) for i in range(starting_idx, starting_idx + len(raw_data))],
            [line.strip() for line in raw_data],
            s_embeddings,
        ]

        if self.milvus_collection is None:
            self.logger.error("Collection is empty.")
            raise Exception("Collection is empty.")
        insert_result = self.milvus_collection.insert(entities)

        # After final entity is inserted, it is best to call flush to have no growing segments left in memory
        self.milvus_collection.flush()

        return insert_result

    def connect(self) -> None:
        """
        Connects to the milvus db.
        """
        try:
            connections.connect(self.alias, self.host, self.port)
        except Exception as e:
            self.logger.error(
                f"Something went wrong while trying to connect to Milvus: {e}"
            )
        else:
            self.logger.info("Successfully connected to Milvus.")

    def search(
        self, text: str, limit: int, output_fields: List[str]
    ) -> pymilvus.orm.search.Hits:
        """
        Receives a text and searches a collection for similar texts, based on their embeddings representions.

        :param text: str
            The text to search in the collection
        :param limit: int
            The number of similar results to be returned
        :param output_fields: List[str]
            The collection fields to be returned
        :return: pymilvus.orm.search.Hits
            The most similar to the text collection entries
        """
        text_embedding = self.model.encode(text)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        start_time = time.time()

        if self.milvus_collection is None:
            self.logger.error("Collection is empty.")
            raise Exception("Collection is empty.")

        result = self.milvus_collection.search(
            [text_embedding],
            "embeddings",
            search_params,
            limit=limit,
            output_fields=output_fields,
        )
        end_time = time.time()

        for hits in result:
            for hit in hits:
                self.logger.info(f"hit: {hit}, id: {hit.entity.get('pk')}")
        self.logger.info(f"Search latency: {end_time - start_time}")
        return hits

    def drop_db(self) -> None:
        """
        Drops a collection
        """
        try:
            utility.drop_collection(self.collection_name)
        except Exception as e:
            self.logger.error(
                f"Something went wrong while trying to drop the collection: {e}"
            )
        else:
            self.logger.info(f"Collection {self.collection_name} successfully dropped.")
