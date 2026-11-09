import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from jina_key_failover import request_with_jina_key_failover


class JinaKeyFailoverTests(unittest.TestCase):
    def run_request(self, responses):
        calls = []
        def request(headers):
            calls.append(headers.get("Authorization"))
            return responses[len(calls) - 1]
        with patch.dict("os.environ", {"JINA_API_KEYS": "a,b,c", "JINA_API_KEY": "a,b,c"}):
            result = request_with_jina_key_failover(
                request, url="https://r.jina.ai/http://example.com",
                headers={"Authorization": "Bearer a"},
            )
        return result, calls

    def test_balance_error_switches_until_success(self):
        exhausted = {"status_code": 402, "text": '{"name":"InsufficientBalanceError"}'}
        success = {"status_code": 200, "text": "page"}
        result, calls = self.run_request([exhausted, exhausted, success])
        self.assertEqual(calls, ["Bearer a", "Bearer b", "Bearer c"])
        self.assertIs(result, success)

    def test_other_errors_do_not_switch(self):
        for status, text in [(401, '{}'), (429, '{}'), (502, '{}'), (402, '{"name":"OtherError"}')]:
            with self.subTest(status=status, text=text):
                response = {"status_code": status, "text": text}
                result, calls = self.run_request([response])
                self.assertEqual(calls, ["Bearer a"])
                self.assertIs(result, response)

    def test_all_exhausted_keys_are_tried_once(self):
        exhausted = {"status_code": 402, "text": '{"name":"InsufficientBalanceError"}'}
        result, calls = self.run_request([exhausted] * 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 3)
        self.assertIs(result, exhausted)


if __name__ == "__main__":
    unittest.main()
