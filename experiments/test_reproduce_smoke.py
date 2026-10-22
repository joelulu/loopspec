import json
import tempfile
import unittest
from pathlib import Path

from experiments.reproduce_smoke import consume_stream, read_metrics


def stream(events):
    return [f"data: {json.dumps(e)}\n".encode() for e in events]+[b"data: [DONE]\n"]


class SmokeProtocolTests(unittest.TestCase):
    def test_cumulative_ids(self):
        result=consume_stream(stream([
            {"text":"a","output_ids":[1],"meta_info":{"completion_tokens":1}},
            {"text":"ab","output_ids":[1,2],"meta_info":{"completion_tokens":2}},
        ]))
        self.assertEqual(result["token_ids"],[1,2])

    def test_incremental_ids(self):
        result=consume_stream(stream([
            {"text":"a","output_ids":[1],"meta_info":{"completion_tokens":1}},
            {"text":"bc","output_ids":[2,3],"meta_info":{"completion_tokens":3}},
        ]))
        self.assertEqual(result["token_ids"],[1,2,3])
        self.assertEqual(result["text"],"abc")

    def test_missing_ids_not_retokenized(self):
        result=consume_stream(stream([{"text":"abc","meta_info":{"completion_tokens":3}}]))
        self.assertIsNone(result["token_ids"])

    def test_truncated_stream_fails(self):
        with self.assertRaises(RuntimeError):
            consume_stream([b'data: {"text":"a","meta_info":{"completion_tokens":1}}\n'])

    def test_misaligned_ids_fail(self):
        with self.assertRaises(RuntimeError):
            consume_stream(stream([{"output_ids":[1,2],"meta_info":{"completion_tokens":5}}]))

    def test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"server.log"
            path.write_text('log\n[server] generation_metrics {"decode_tokens":7,"decode_seconds":0.2}\n')
            self.assertEqual(read_metrics(path)[0]["decode_tokens"],7)


if __name__ == "__main__":
    unittest.main()
