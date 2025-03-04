import pytest
from unittest.mock import MagicMock, patch
from app.tools.outline_generator.core import executor
from app.tools.outline_generator.tools import OutlineGenerator, Outlines
from app.services.schemas import OutlineGeneratorInput
from app.api.error_utilities import LoaderError, ToolExecutorError
from langchain_core.documents import Document

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
        assert isinstance(result, Outlines)
    

# @patch('app.utils.document_loaders.get_docs')
# def test_executor_document_loading_error(mock_get_docs):
#     """Test the executor function when document loading fails."""
#     # Mock the get_docs function to raise LoaderError
#     mock_get_docs.side_effect = LoaderError("Failed to load document")
    
#     with pytest.raises(ToolExecutorError):
#         executor(
#             n_slides=base_attributes["n_slides"],
#             topic=base_attributes["topic"],
#             instructional_level=base_attributes["instructional_level"],
#             file_upload_url=base_attributes["file_upload_url"],
#             file_upload_type=base_attributes["file_upload_type"],
#             lang=base_attributes["lang"],
#             verbose=False
#         )

# # Test compile_with_context method
# @patch('langchain_chroma.Chroma.from_documents')
# def test_outline_generator_compile_with_context(mock_from_documents, monkeypatch):
#     """Test the compile_with_context method."""
#     # Mock the vectorstore_class.from_documents method
#     mock_vectorstore = MagicMock()
#     mock_retriever = MagicMock()
#     mock_vectorstore.as_retriever.return_value = mock_retriever
#     mock_from_documents.return_value = mock_vectorstore
    
#     generator = OutlineGenerator(args=mock_args, verbose=False)
#     documents = [Document(page_content="Sample document content")]
#     chain = generator.compile_with_context(documents)
    
#     assert chain is not None
#     assert generator.vectorstore is not None
#     assert generator.retriever is not None
#     assert generator.runner is not None

# # Test compile_without_context method
# def test_outline_generator_compile_without_context():
#     """Test the compile_without_context method."""
#     generator = OutlineGenerator(args=mock_args, verbose=False)
#     chain = generator.compile_without_context()
    
#     assert chain is not None

# # Test generate_outline method with documents
# def test_outline_generator_generate_outline_with_docs(monkeypatch):
#     """Test the generate_outline method with documents."""
#     mock_chain = MagicMock()
#     mock_output = Outlines(outlines=["Slide 1: Introduction to Python", "Slide 2: Variables and Data Types"])
#     mock_chain.invoke.return_value = mock_output
    
#     def mock_compile(self, documents):
#         return mock_chain
    
#     # Use monkeypatch to replace the compile_with_context method
#     monkeypatch.setattr(OutlineGenerator, "compile_with_context", mock_compile)
    
#     generator = OutlineGenerator(args=mock_args, verbose=False)
#     documents = [Document(page_content="Sample document content")]
#     result = generator.generate_outline(documents)
    
#     assert isinstance(result, Outlines)
#     assert len(result.outlines) == 2
#     assert result.outlines[0] == "Slide 1: Introduction to Python"

# # Test generate_outline method without documents
# def test_outline_generator_generate_outline_without_docs(monkeypatch):
#     """Test the generate_outline method without documents."""
#     mock_chain = MagicMock()
#     mock_output = Outlines(outlines=["Slide 1: Introduction to Python", "Slide 2: Variables and Data Types"])
#     mock_chain.invoke.return_value = mock_output
    
#     def mock_compile(self):
#         return mock_chain
    
#     # Use monkeypatch to replace the compile_without_context method
#     monkeypatch.setattr(OutlineGenerator, "compile_without_context", mock_compile)
    
#     generator = OutlineGenerator(args=mock_args, verbose=False)
#     result = generator.generate_outline(None)
    
#     assert isinstance(result, Outlines)
#     assert len(result.outlines) == 2
#     assert result.outlines[0] == "Slide 1: Introduction to Python"

# # Test OutlineGenerator with invalid arguments
# def test_outline_generator_invalid_args():
#     """Test OutlineGenerator with invalid arguments."""
#     invalid_args = OutlineGeneratorInput(
#         n_slides=base_attributes["n_slides"],
#         topic=None,  # Topic is required
#         instructional_level=base_attributes["instructional_level"],
#         file_upload_url=base_attributes["file_upload_url"],
#         file_upload_type=base_attributes["file_upload_type"],
#         lang=base_attributes["lang"]
#     )
    
#     with pytest.raises(ValueError):
#         OutlineGenerator(args=invalid_args, verbose=False)