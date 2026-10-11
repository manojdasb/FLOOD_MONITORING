import importlib.util
import pathlib
import unittest

APP_PATH = pathlib.Path(__file__).resolve().parents[1] / "flood_system" / "app.py"
SPEC = importlib.util.spec_from_file_location("flood_app", APP_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(module)


class BackendHealthTests(unittest.TestCase):
    def test_backend_healthy_detects_ok(self):
        original_get = module.httpx.get

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"status": "ok"}

        def fake_get(url, timeout):
            self.assertEqual(url, "http://localhost:8000/health")
            self.assertEqual(timeout, 2.0)
            return FakeResponse()

        module.httpx.get = fake_get
        try:
            self.assertTrue(module.backend_healthy("http://localhost:8000"))
        finally:
            module.httpx.get = original_get


if __name__ == "__main__":
    unittest.main()
