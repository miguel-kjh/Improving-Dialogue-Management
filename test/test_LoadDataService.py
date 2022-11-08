from unittest import TestCase

import pandas as pd

from service.LoadDataService import LoadDataService


class TestLoadDataService(TestCase):

    config = {
        'dataset': {
            'DB_name': 'SGD',
            'name': 'SGD_dataset',
            'domain': 'TINY'
        },
        'database': [
            {
                'path': 'mongodb://localhost:27017/'
            }
        ]
    }

    def test_run(self):
        lds = LoadDataService(self.config)
        df = lds.run()
        self.assertTrue(df is not None)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(len(df) > 0)
