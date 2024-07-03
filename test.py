import unittest

from LazadaIDReviews.config.configuration import ConfigurationManager
from LazadaIDReviews.components.unit_testing import UnitTesting

config = ConfigurationManager()
unit_test_config = config.get_unit_test_config()
test = UnitTesting(config=unit_test_config)
test.set_request_body()

class TestPredictFunction(unittest.TestCase):
    def test_input_output_size(self):
        self.assertEqual(
            test.get_output_length(), 
            len(test.get_request_body_value())
        )
    
    def test_output_type(self):
        self.assertEqual(
            test.is_output_type_list(), 
            True
        )
    
    def test_output_type_consistency(self):
        self.assertEqual(
            test.is_output_type_consistent(), 
            True
        )

if __name__ == '__main__':
    unittest.main()