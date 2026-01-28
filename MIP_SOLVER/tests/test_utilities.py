"""
Unit tests for the utilities module (solver/utilities.py).

Tests the logger setup functionality.
"""
import pytest
import logging
from solver.utilities import setup_logger


class TestSetupLogger:
    """Tests for the setup_logger function."""

    def test_logger_creation(self):
        """Test that setup_logger returns a logger instance."""
        logger = setup_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "MIPSolver"

    def test_logger_returns_same_instance(self):
        """Test that setup_logger returns the same logger instance on multiple calls."""
        logger1 = setup_logger()
        logger2 = setup_logger()

        assert logger1 is logger2

    def test_logger_can_log_debug(self):
        """Test that the logger can log debug messages without errors."""
        logger = setup_logger()

        # This should not raise any exceptions
        try:
            logger.debug("Debug test message")
        except Exception as e:
            pytest.fail(f"Logger raised exception on debug: {e}")

    def test_logger_can_log_info(self):
        """Test that the logger can log info messages without errors."""
        logger = setup_logger()

        try:
            logger.info("Info test message")
        except Exception as e:
            pytest.fail(f"Logger raised exception on info: {e}")

    def test_logger_can_log_warning(self):
        """Test that the logger can log warning messages without errors."""
        logger = setup_logger()

        try:
            logger.warning("Warning test message")
        except Exception as e:
            pytest.fail(f"Logger raised exception on warning: {e}")

    def test_logger_can_log_error(self):
        """Test that the logger can log error messages without errors."""
        logger = setup_logger()

        try:
            logger.error("Error test message")
        except Exception as e:
            pytest.fail(f"Logger raised exception on error: {e}")

    def test_setup_logger_idempotent(self):
        """Test that calling setup_logger multiple times is safe."""
        # Call multiple times - should not throw errors
        for _ in range(5):
            logger = setup_logger()
            assert logger.name == "MIPSolver"

    def test_logger_has_correct_name(self):
        """Test that the logger has the correct name."""
        logger = setup_logger()
        assert logger.name == "MIPSolver"

    def test_logger_not_none(self):
        """Test that setup_logger never returns None."""
        logger = setup_logger()
        assert logger is not None


class TestLoggerFunctional:
    """Functional tests for the logger in realistic scenarios."""

    def test_logger_used_from_multiple_modules(self):
        """Test that the logger works when used from multiple imports."""
        from solver.utilities import setup_logger as setup1
        from solver.utilities import setup_logger as setup2

        logger1 = setup1()
        logger2 = setup2()

        # Both should return the same logger
        assert logger1 is logger2
        assert logger1.name == "MIPSolver"

    def test_logger_format_method_exists(self):
        """Test that logger has standard logging methods."""
        logger = setup_logger()

        # Standard logging methods should exist
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)
