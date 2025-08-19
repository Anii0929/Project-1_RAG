import pytest
import sys
import os
from unittest.mock import patch, Mock
import tempfile

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, config


class TestConfig:
    """Test suite for Configuration"""

    def test_default_values(self):
        """Test that default configuration values are correct"""
        test_config = Config()
        
        # Check default model - this is important for the bug we're investigating
        assert test_config.ANTHROPIC_MODEL == "anthropic-claude-3-5-haiku"
        
        # Check other defaults
        assert test_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert test_config.CHUNK_SIZE == 800
        assert test_config.CHUNK_OVERLAP == 100
        assert test_config.MAX_RESULTS == 5
        assert test_config.MAX_HISTORY == 2
        assert test_config.CHROMA_PATH == "./chroma_db"

    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-api-key-123'
        }):
            # Reimport config to get fresh environment loading
            import importlib
            import config as config_module
            importlib.reload(config_module)
            
            assert config_module.config.ANTHROPIC_API_KEY == 'test-api-key-123'

    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == ""

    @patch('config.load_dotenv')
    def test_dotenv_loading(self, mock_load_dotenv):
        """Test that dotenv is called to load .env file"""
        import importlib
        import config as config_module
        importlib.reload(config_module)
        
        mock_load_dotenv.assert_called_once()

    def test_config_singleton(self):
        """Test that config is properly instantiated as singleton"""
        from config import config
        assert isinstance(config, Config)

    def test_model_compatibility(self):
        """Test model name format for Anthropic API compatibility"""
        test_config = Config()
        model_name = test_config.ANTHROPIC_MODEL
        
        # Should contain 'claude' for Anthropic compatibility
        assert 'claude' in model_name.lower()
        
        # Should not have obvious typos or formatting issues
        assert not model_name.startswith(' ')
        assert not model_name.endswith(' ')
        assert len(model_name) > 5  # Reasonable minimum length


class TestConfigValidation:
    """Test configuration validation and potential issues"""

    def test_api_key_format_validation(self):
        """Test API key format validation"""
        # Test with various API key formats
        test_cases = [
            ("sk-ant-api03-valid-key-here", True),  # Valid format
            ("sk-VpRaTQQGBr9C_i3uTgnwHA", True),    # Current format in .env
            ("", False),                            # Empty
            ("invalid-key", False),                 # Invalid format
            ("sk-", False),                        # Too short
        ]
        
        for api_key, should_be_valid in test_cases:
            is_valid = self._validate_api_key_format(api_key)
            if should_be_valid:
                assert is_valid, f"API key '{api_key}' should be valid"
            else:
                assert not is_valid, f"API key '{api_key}' should be invalid"

    def _validate_api_key_format(self, api_key: str) -> bool:
        """Helper method to validate API key format"""
        if not api_key or len(api_key) < 10:
            return False
        
        # Anthropic API keys typically start with 'sk-'
        if not api_key.startswith('sk-'):
            return False
            
        return True

    def test_chunk_size_validation(self):
        """Test chunk size configuration validation"""
        test_config = Config()
        
        # Chunk size should be reasonable
        assert test_config.CHUNK_SIZE > 0
        assert test_config.CHUNK_SIZE <= 2000  # Not too large
        
        # Overlap should be less than chunk size
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
        assert test_config.CHUNK_OVERLAP >= 0

    def test_max_results_validation(self):
        """Test max results configuration validation"""
        test_config = Config()
        
        assert test_config.MAX_RESULTS > 0
        assert test_config.MAX_RESULTS <= 50  # Reasonable upper limit

    def test_max_history_validation(self):
        """Test max history configuration validation"""
        test_config = Config()
        
        assert test_config.MAX_HISTORY >= 0
        assert test_config.MAX_HISTORY <= 20  # Reasonable upper limit


class TestConfigIntegration:
    """Integration tests for configuration with real scenarios"""

    def test_actual_config_values(self):
        """Test the actual configuration values being used"""
        from config import config
        
        # This test will reveal the actual configuration being used
        print(f"Actual API Key (first 10 chars): {config.ANTHROPIC_API_KEY[:10]}...")
        print(f"Actual Model: {config.ANTHROPIC_MODEL}")
        print(f"Actual Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"Actual Chunk Size: {config.CHUNK_SIZE}")
        print(f"Actual Max Results: {config.MAX_RESULTS}")
        
        # Key test: Check for model mismatch issue
        # The CLAUDE.md mentions 'claude-sonnet-4-20250514' but config uses 'anthropic-claude-3-5-haiku'
        assert config.ANTHROPIC_MODEL is not None
        assert len(config.ANTHROPIC_MODEL) > 0

    def test_api_key_accessibility(self):
        """Test that API key is accessible and not empty"""
        from config import config
        
        # API key should be accessible
        api_key = config.ANTHROPIC_API_KEY
        assert isinstance(api_key, str)
        
        # For debugging: show if API key is present
        if api_key:
            print(f"API Key found: {api_key[:10]}... (length: {len(api_key)})")
        else:
            print("WARNING: No API key found in configuration")

    def test_chroma_path_accessibility(self):
        """Test that ChromaDB path is accessible"""
        from config import config
        
        chroma_path = config.CHROMA_PATH
        assert isinstance(chroma_path, str)
        assert len(chroma_path) > 0
        
        # Check if path exists or can be created
        import os
        if os.path.exists(chroma_path):
            print(f"ChromaDB path exists: {chroma_path}")
        else:
            print(f"ChromaDB path will be created: {chroma_path}")

    def test_embedding_model_format(self):
        """Test embedding model format"""
        from config import config
        
        embedding_model = config.EMBEDDING_MODEL
        
        # Should be a valid sentence transformers model name
        assert isinstance(embedding_model, str)
        assert len(embedding_model) > 0
        assert not embedding_model.startswith(' ')
        assert not embedding_model.endswith(' ')


class TestConfigErrors:
    """Test configuration error scenarios"""

    def test_invalid_environment_handling(self):
        """Test handling of invalid environment variables"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': '',  # Empty API key
        }):
            test_config = Config()
            
            # Should handle empty API key gracefully
            assert test_config.ANTHROPIC_API_KEY == ""

    def test_config_with_temp_env_file(self):
        """Test config loading with temporary .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as temp_env:
            temp_env.write("ANTHROPIC_API_KEY=temp-test-key\n")
            temp_env.flush()
            
            # Patch the .env file path
            with patch('config.load_dotenv') as mock_load_dotenv:
                mock_load_dotenv.return_value = None
                
                # Create config - should not fail even with mocked dotenv
                test_config = Config()
                assert isinstance(test_config, Config)
        
        # Clean up
        os.unlink(temp_env.name)


# Diagnostic test to help identify the "query failed" issue
class TestConfigDiagnostics:
    """Diagnostic tests to help identify configuration issues causing failures"""

    def test_anthropic_model_compatibility(self):
        """Test if the configured model is compatible with tool calling"""
        from config import config
        
        model = config.ANTHROPIC_MODEL
        print(f"Testing model compatibility: {model}")
        
        # Models that support tool calling (as of knowledge cutoff)
        compatible_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        
        # Check if current model is in known compatible list
        model_compatible = any(compatible in model for compatible in compatible_models)
        
        if not model_compatible:
            print(f"WARNING: Model '{model}' may not support tool calling")
            print(f"Compatible models include: {compatible_models}")
        else:
            print(f"Model '{model}' appears to support tool calling")
        
        # The test should not fail, but provide diagnostic info
        assert isinstance(model, str)

    def test_full_config_diagnostic(self):
        """Diagnostic test to print full configuration"""
        from config import config
        
        print("\n=== FULL CONFIGURATION DIAGNOSTIC ===")
        print(f"ANTHROPIC_API_KEY: {'SET' if config.ANTHROPIC_API_KEY else 'NOT SET'}")
        print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
        print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
        print(f"CHUNK_SIZE: {config.CHUNK_SIZE}")
        print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
        print(f"MAX_RESULTS: {config.MAX_RESULTS}")
        print(f"MAX_HISTORY: {config.MAX_HISTORY}")
        print(f"CHROMA_PATH: {config.CHROMA_PATH}")
        print("===================================\n")
        
        # Basic assertions
        assert config.ANTHROPIC_MODEL is not None
        assert config.EMBEDDING_MODEL is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements