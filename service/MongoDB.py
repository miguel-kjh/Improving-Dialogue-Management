from copy import copy

from git import Object
from service.InputService import InputService
from service.OutputService import OutputService
from view.Logger import Logger
from utils.mongo_db_utils import LOCAL_SERVER

try:
    from pymongo import MongoClient
    import pandas as pd
except Exception as e:
    print("Some Modules are Missing in MongoDB.py")


class MongoDB(InputService, OutputService):

    def __init__(self, db_name: str, local_server: str):
        self.dBName = db_name
        self.client = MongoClient(local_server)
        self.DB = self.client[self.dBName]

    def load(self, path: str, to_pandas: bool = True) -> object:
        collection = self.DB[path]
        data = collection.find({})
        if to_pandas:
            df = pd.DataFrame(list(data))
        else:
            df = list(data)
        Logger.info("Data has been Loaded from Mongo DB Server .... ")
        return df

    def save(self, df: Object, path: str) -> None:
        collection = self.DB[path]

        if type(df) is pd.DataFrame:
            data = df.to_dict('records')
        else:
            data = df

        if collection.count_documents({}) > 0:
            Logger.info("Dropping last data from Mongo DB Server .... ")
            collection.drop()

        try:
            collection.insert_many(data, ordered=False)
        except Exception as _:
            for key, record in data.items():
                record['Dialogue_id'] = key
                collection.insert_one(record)

        Logger.info("All the Data has been Exported to Mongo DB Server .... ")
