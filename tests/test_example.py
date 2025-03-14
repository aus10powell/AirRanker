import pytest
import logging

@pytest.mark.unit
class TestExample:
    def test_basic_assertion(self):
        x = 1
        y = 1
        logging.info(f"Testing if {x} equals {y}")
        assert x == y

    @pytest.mark.slow
    def test_slow_operation(self):
        import time
        logging.info("Starting slow operation")
        time.sleep(1)  # Simulate slow operation
        assert True

    @pytest.mark.integration
    def test_integration_example(self):
        # This is where you'd test integration with other components
        print("This print statement will be visible due to --capture=no")
        assert True

    def test_with_locals(self):
        x = "actual"
        expected = "expected"
        # This will show local variables due to --showlocals
        assert x == expected 