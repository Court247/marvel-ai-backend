import pytest
from unittest.mock import MagicMock, patch
from app.tools.presentation_generator.outline_generator.core import executor
from app.tools.presentation_generator.outline_generator.tools import OutlineGenerator, Outlines
from app.services.schemas import OutlineGeneratorInput
from app.api.error_utilities import LoaderError, ToolExecutorError
from langchain_core.documents import Document
from unittest.mock import Mock
# Base test attributes
base_attributes = {
    "n_slides": 2,
    "topic": "Introduction to Python Programming",
    "instructional_level": "beginner",
    "file_upload_url": "",
    "file_upload_type": "",
    "lang": "en"
}

# Mock OutlineGeneratorInput
mock_args = OutlineGeneratorInput(
    n_slides=base_attributes["n_slides"],
    topic=base_attributes["topic"],
    instructional_level=base_attributes["instructional_level"],
    file_upload_url=base_attributes["file_upload_url"],
    file_upload_type=base_attributes["file_upload_type"],
    lang=base_attributes["lang"]
)

# Test OutlineGenerator class initialization
def test_outline_generator_init():
    """Test initialization of OutlineGenerator."""
    generator = OutlineGenerator(args=mock_args, verbose=False)
    assert generator.args is not None
    assert generator.verbose is False
    assert generator.vectorstore is None
    assert generator.retriever is None
    assert generator.runner is None

# Test the executor function (integration test)

def test_executor_normal_operation():
    """Test the executor function with valid inputs."""
    # Set up mock returns
    
    result = executor(
        n_slides=base_attributes["n_slides"],
        topic=base_attributes["topic"],
        instructional_level=base_attributes["instructional_level"],
        file_upload_url="",
        file_upload_type="",
        lang=base_attributes["lang"],
        verbose=False
    )
    # Check if the result is a dictionary instead of an Outlines instance
    assert isinstance(result, dict), f"Expected dict, but got {type(result)}"

    # Validate the structure of the dictionary
    assert "outlines" in result, "Key 'outlines' not found in response"
    assert isinstance(result["outlines"], list), "Expected 'outlines' to be a list"
    

def test_executor_missing_required_inputs():
    """Test the executor function with missing required inputs."""
    with pytest.raises(ValueError):
        result =  executor(
                n_slides=None,
                topic=None,
                instructional_level=base_attributes["instructional_level"],
                file_upload_url=base_attributes["file_upload_url"],
                file_upload_type=base_attributes["file_upload_type"],
                lang=base_attributes["lang"],
                verbose=False
            )
# Test OutlineGenerator with invalid arguments
def test_outline_generator_init_missing_params():
    """Test initialization of OutlineGenerator with missing parameters."""

    with pytest.raises(ValueError, match="Topic must be provided"):
        OutlineGenerator(args=Mock(topic=None, lang="en"))
    
    with pytest.raises(ValueError, match="Language must be provided"):
        OutlineGenerator(args=Mock(topic="Test", lang=None))


def test_outline_generator_compile_without_context():
    """Test compilation of pipeline without context."""
    args = OutlineGeneratorInput(
        n_slides=3,
        topic="Machine Learning",
        instructional_level="Advanced",
        file_upload_url="",
        file_upload_type="",
        lang="en"
    )
    
    generator = OutlineGenerator(args=args)
    chain = generator.compile_without_context()
    
    assert chain is not None

def test_outlines_model():
    """Test the Outlines Pydantic model."""
    outlines = Outlines(
        outlines=[
            "Introduction to Cybersecurity",
            "Types of Cyber Threats",
            "Basic Security Practices"
        ]
    )
    
    assert len(outlines.outlines) == 3
    assert all(isinstance(outline, str) for outline in outlines.outlines)