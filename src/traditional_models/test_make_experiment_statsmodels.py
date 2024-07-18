import unittest
from unittest.mock import MagicMock

from mlflow.entities import FileInfo, Metric, Param, RunData, RunTag
from mlflow.tracking import MlflowClient


class TestFetchLoggedData(unittest.TestCase):
    def test_fetch_logged_data(self):
        # Create a mock MlflowClient
        client = MlflowClient()
        client.get_run = MagicMock(return_value=RunData(
            params=[Param(key='param1', value='value1')],
            metrics=[Metric(key='metric1', value=0.5)],
            tags=[RunTag(key='tag1', value='value1')],
            artifacts=[FileInfo(path='artifact1'), FileInfo(path='artifact2')]
        ))

        # Call the function under test
        params, metrics, tags, artifacts = fetch_logged_data('run_id', client)

        # Assert the expected results
        self.assertEqual(params, {'param1': 'value1'})
        self.assertEqual(metrics, {'metric1': 0.5})
        self.assertEqual(tags, {'tag1': 'value1'})
        self.assertEqual(artifacts, ['artifact1', 'artifact2'])

if __name__ == '__main__':
    unittest.main()