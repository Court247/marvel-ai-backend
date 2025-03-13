import pytest
from unittest.mock import patch
from app.tools.syllabus_generator.tools import (
    SyllabusGeneratorPipeline, 
    CompilePipelineError, 
    ChainBuilder, 
    resume_course_content, 
    PromptFactory, 
    ParserFactory, 
    SyllabusGenerator, 
    SyllabusRequestArgs, 
    SyllabusGeneratorArgsModel, 
    OutputValidationError
)
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from app.tools.syllabus_generator.tools import SyllabusGenerator, SyllabusRequestArgs, SyllabusGeneratorArgsModel
from unittest.mock import patch, MagicMock


@pytest.fixture
def pipeline():
    return SyllabusGeneratorPipeline()

@pytest.fixture
def fallbacks_list(pipeline):
    return ChainBuilder(None, {}).create_fallback("test_section")

@pytest.fixture
def chains(pipeline):
    return pipeline.compile()

@pytest.fixture
def sections_chains(chains):
    return {**chains["sequential"], **chains["parallel"].steps__}

@pytest.fixture
def fallback(fallbacks_list):
    return fallbacks_list[0]

### Retrive the section chains

def test_create_section_fallback_returns_runnable_lambda(fallback):
    """Test that create_section_fallback returns a RunnableLambda object."""
    assert isinstance(fallback, RunnableLambda)

def test_course_information_chain_invoke_with_fallback(sections_chains):
    """Test that the course_information section chain invokes the fallback when the chain fails."""

    course_information_chain = sections_chains["course_information"]
    with patch(
            "langchain_core.prompts.base.BasePromptTemplate.invoke", 
            return_value={"output": "success"}
        ) as parser_mock, \
        patch(
            "langchain_google_genai.llms.GoogleGenerativeAI.invoke",
            side_effect=ValueError("Simulated failure")
        ) as mock_generate:

        result = course_information_chain.invoke({"user_query":"Test input"})
        assert mock_generate.call_count == 1
        assert result["status"] == "failed"
        assert result["section"] == "CourseInformation"
        assert result["fallback"] is True
        assert result["error"] == "Simulated failure"

def test_compile_uses_fallback(sections_chains):
    """Test that the compile method sets up chains with fallbacks."""

    # Verify the course_information chain has fallback configured
    assert 'course_information' in sections_chains
    assert hasattr(sections_chains['course_information'], 'fallbacks')
    assert len(sections_chains['course_information'].fallbacks) > 0

    assert 'course_description_objectives' in sections_chains
    assert hasattr(sections_chains['course_description_objectives'], 'fallbacks')
    assert len(sections_chains['course_description_objectives'].fallbacks) > 0

    assert 'course_content' in sections_chains
    assert hasattr(sections_chains['course_content'], 'fallbacks')
    assert len(sections_chains['course_content'].fallbacks) > 0

    assert 'policies_procedures' in sections_chains
    assert hasattr(sections_chains['policies_procedures'], 'fallbacks')
    assert len(sections_chains['policies_procedures'].fallbacks) > 0

    assert 'assessment_grading_criteria' in sections_chains
    assert hasattr(sections_chains['assessment_grading_criteria'], 'fallbacks')
    assert len(sections_chains['assessment_grading_criteria'].fallbacks) > 0

    assert 'learning_resources' in sections_chains
    assert hasattr(sections_chains['learning_resources'], 'fallbacks')
    assert len(sections_chains['learning_resources'].fallbacks) > 0

    assert 'course_schedule' in sections_chains
    assert hasattr(sections_chains['course_schedule'], 'fallbacks')
    assert len(sections_chains['course_schedule'].fallbacks) > 0

    assert 'course_schedule' in sections_chains
    assert hasattr(sections_chains['course_schedule'], 'fallbacks')
    assert len(sections_chains['course_schedule'].fallbacks) > 0

def test_chain_with_fallback_integration(fallbacks_list):
    """Test the fallback integration with a chain that fails."""
    
    # Use a real Runnable that will fail instead of a mock
    def failing_function(input_data: dict, *args, **kwargs):
        raise ValueError("Simulated failure")
    
    # Add fallback to the real runnable
    failing_chain = RunnableLambda(failing_function)
    chain_with_fallback = failing_chain.with_fallbacks(fallbacks_list, exception_key = "error")
    
    # Execute the chain - this should trigger the fallback
    result = chain_with_fallback.invoke({"query": "Test input"})
    
    # Verify fallback was used
    assert result["status"] == "failed"
    assert result["section"] == "test_section"
    assert result["fallback"] is True
    assert result["error"] == "Simulated failure"

def test_section_fallback_handles_basic_input(fallback):
    """Test the fallback function with a basic input."""
    result = fallback.invoke("Test input")
    
    assert result["status"] == "failed"
    assert result["error"] == "Failed to generate test_section section."
    assert result["section"] == "test_section" 
    assert result["fallback"] is True

def test_section_chain_invoke_with_fallback(fallbacks_list):
    """Test that the section chain invokes the fallback when the chain fails."""

    # Create a success function that returns the expected dictionary
    def success_function(input_data, *args, **kwargs):
        return {
            "status": "success",
            "result": "Success",
            "section": "test_section",
            "fallback": False
        }
    
    # Use RunnableLambda instead of custom Runnable class
    success_chain = RunnableLambda(success_function)
    chain_with_success = success_chain.with_fallbacks(fallbacks_list, exception_key = "error" )
    
    # Check that success result doesn't trigger fallback
    success_result = chain_with_success.invoke({"input": "Test input"})

    # When primary succeeds, result should be the primary's return value
    assert success_result["status"] == "success"
    assert success_result["fallback"] is False
    assert success_result["result"] == "Success"

# ------------------------------------------
## Test Pipeline Compilation
def test_compile_returns_dict_with_correct_keys(chains):
    """Test that compile returns a dictionary with sequential and parallel keys."""
    assert isinstance(chains, dict)
    assert "sequential" in chains
    assert "parallel" in chains

def test_course_information_chain_invoke_with_success(sections_chains):
    """Test that the course_information section chain invokes the fallback when the chain fails."""
    course_information_chain = sections_chains["course_information"]
    with patch(
            "langchain_core.prompts.base.BasePromptTemplate.invoke", 
            return_value={"output": "Parser success"}
        ) as prompt_mock, \
        patch(
            "langchain_google_genai.llms.GoogleGenerativeAI.invoke",
            return_value={"output": "Model success"}
        ) as model_mock , \
        patch(
            "langchain_core.output_parsers.json.JsonOutputParser.invoke", 
            return_value={"output": "Final output: success"}
        ) as parser_mock:
        
        result = course_information_chain.invoke({"user_query":"Test input"})
        print(result)
        assert prompt_mock.call_count == 1
        assert model_mock.call_count == 1
        assert parser_mock.call_count == 1
        assert result["output"] == "Final output: success"

def test_compile_chains_structure(chains):
    """Test that compile returns the correct sequential chains."""
    sequential = chains["sequential"]

    # Verify all expected sequential chains exist
    assert "course_information" in sequential
    assert "course_description_objectives" in sequential
    assert "course_content" in sequential
    assert "policies_procedures" in sequential

    # Check each chain is a runnable
    for chain_name, chain in sequential.items():
        assert hasattr(chain, "invoke"), f"{chain_name} is not a runnable"

def test_compile_parallel_pipeline_structure(pipeline):
    """Test that compile returns the correct parallel pipeline structure."""
    result = pipeline.compile()
    parallel = result["parallel"].steps__

    assert "assessment_grading_criteria" in parallel
    assert "learning_resources" in parallel
    assert "course_schedule" in parallel

    # Check each branch is a runnable
    for branch_name, branch in parallel.items():
        assert hasattr(branch, "invoke"), f"{branch_name} is not a runnable"

def test_compile_handles_errors(pipeline):
    """Test that compile handles errors properly."""
    with patch("app.tools.syllabus_generator.tools.ParserFactory.create_parsers", 
            side_effect=Exception("Test error")):
        with pytest.raises(CompilePipelineError):
            pipeline.compile()

def test_compile_with_mocked_dependencies():
    """Test compile with mocked dependencies to verify integration."""
    # Mock all necessary dependencies to isolate the compile method
    with patch("app.tools.syllabus_generator.tools.ParserFactory.create_parsers") as mock_parsers, \
        patch("app.tools.syllabus_generator.tools.ChainBuilder") as mock_builder:

        # Configure mocks
        mock_chain = MagicMock()
        mock_builder_instance = mock_builder.return_value
        mock_builder_instance.build_chain_with_fallback.return_value = mock_chain

        # Create pipeline and compile
        pipeline = SyllabusGeneratorPipeline()
        result = pipeline.compile()

        # Verify parser factory was called
        mock_parsers.assert_called_once()

        # Verify ChainBuilder was instantiated with the right arguments
        mock_builder.assert_called_once()

        assert mock_builder_instance.build_chain_with_fallback.call_count == 7
        # Verify the compiled result has the right structure
        assert result["sequential"]["course_information"] == mock_chain
        assert result["sequential"]["course_description_objectives"] == mock_chain
        assert result["sequential"]["course_content"] == mock_chain
        assert result["sequential"]["policies_procedures"] == mock_chain

@pytest.fixture
def sample_syllabus_args():
    return SyllabusGeneratorArgsModel(
        grade_level="High School",
        subject="Mathematics",
        course_description="Introduction to Algebra",
        objectives="Learn basic algebraic concepts",
        required_materials="Textbook, calculator",
        grading_policy="Standard grading scale",
        policies_expectations="Regular attendance required",
        course_outline="Basic algebra concepts",
        additional_notes="None",
        lang="en",
        file_url="",
        file_type=""
    )


@pytest.fixture
def sample_summary():
    return "This is a sample course summary"

@pytest.fixture
def syllabus_request_args(sample_syllabus_args, sample_summary):
    return SyllabusRequestArgs(sample_syllabus_args, sample_summary)

def test_syllabus_request_args_initialization(syllabus_request_args):
    """Test the initialization of SyllabusRequestArgs with default values."""
    print("Testing syllabus_request_args_initialization")
    assert syllabus_request_args._grade_level == "High School"
    assert syllabus_request_args._subject == "Mathematics"
    assert syllabus_request_args._course_description == "Introduction to Algebra"
    assert syllabus_request_args._objectives == "Learn basic algebraic concepts"
    assert syllabus_request_args._required_materials == "Textbook, calculator"
    assert syllabus_request_args._grading_policy == "Standard grading scale"
    assert syllabus_request_args._policies_expectations == "Regular attendance required"
    assert syllabus_request_args._course_outline == "Basic algebra concepts"
    assert syllabus_request_args._additional_notes == "None"
    assert syllabus_request_args._lang == "en"
    assert syllabus_request_args._summary == "This is a sample course summary"

def test_syllabus_request_args_to_dict(syllabus_request_args):
    """Test the to_dict method of SyllabusRequestArgs."""
    print("Testing syllabus_request_args_to_dict")
    result = syllabus_request_args.to_dict()
    assert isinstance(result, dict)
    assert result["grade_level"] == "High School"
    assert result["subject"] == "Mathematics"
    assert result["course_description"] == "Introduction to Algebra"
    assert result["objectives"] == "Learn basic algebraic concepts"
    assert result["required_materials"] == "Textbook, calculator"
    assert result["grading_policy"] == "Standard grading scale"
    assert result["policies_expectations"] == "Regular attendance required"
    assert result["course_outline"] == "Basic algebra concepts"
    assert result["additional_notes"] == "None"
    assert result["lang"] == "en"
    assert result["summary"] == "This is a sample course summary"

def test_prompt_factory_course_information():
    """Test the course_information prompt factory."""
    print("Testing prompt_factory_course_information")
    parser_instructions = "Format as JSON"
    prompt = PromptFactory.course_information(parser_instructions)
    assert isinstance(prompt, PromptTemplate)
    assert "grade_level" in prompt.input_variables
    assert "subject" in prompt.input_variables
    assert "course_description" in prompt.input_variables
    assert "lang" in prompt.input_variables
    assert "summary" in prompt.input_variables
    assert "additional_notes" in prompt.input_variables

def test_parser_factory():
    """Test the parser factory creation."""
    print("Testing parser_factory")
    parsers = ParserFactory.create_parsers()
    assert isinstance(parsers, dict)
    assert "course_information" in parsers
    assert "course_description_objectives" in parsers
    assert "course_content" in parsers
    assert "policies_procedures" in parsers
    assert "assessment_grading_criteria" in parsers
    assert "learning_resources" in parsers
    assert "course_schedule" in parsers
    
    # Verify parser types
    for parser in parsers.values():
        assert isinstance(parser, JsonOutputParser)

def test_resume_course_content():
    """Test the resume_course_content function."""  
    print("Testing resume_course_content")
    course_content = [
        {"unit_time": "weeks", "unit_time_value": 2, "topic": "Introduction"},
        {"unit_time": "weeks", "unit_time_value": 3, "topic": "Basic Concepts"},
        {"unit_time": "days", "unit_time_value": 5, "topic": "Review"}
    ]
    
    result = resume_course_content(course_content)
    assert isinstance(result, dict)
    assert "course_length" in result
    assert "course_topics" in result
    assert "5 weeks" in result["course_length"]
    assert "5 days" in result["course_length"]

def test_syllabus_generator_validate_output():
    """Test the output validation functionality."""
    print("Testing syllabus_generator_validate_output")
    generator = SyllabusGenerator(error_threshold=0.8)
    
    # Test successful validation
    valid_output = {
        "course_information": {"title": "Test Course"},
        "course_description_objectives": {"objectives": ["Objective 1"]},
        "course_content": {"content": ["Content 1"]},
        "policies_procedures": {"policies": ["Policy 1"]},
        "assessment_grading_criteria": {"criteria": ["Criterion 1"]},
        "learning_resources": {"resources": ["Resource 1"]},
        "course_schedule": {"schedule": ["Schedule 1"]}
    }
    
    metadata = generator._validate_output(valid_output)
    assert metadata["status"] == "success"
    assert metadata["error_rate"] == 0
    assert not metadata["error_sections"]
    
    # Test partial failure
    partial_output = {
        "course_information": {"error": "Failed"},
        "course_description_objectives": {"objectives": ["Objective 1"]},
        "course_content": {"content": ["Content 1"]},
        "policies_procedures": {"policies": ["Policy 1"]},
        "assessment_grading_criteria": {"criteria": ["Criterion 1"]},
        "learning_resources": {"resources": ["Resource 1"]},
        "course_schedule": {"schedule": ["Schedule 1"]}
    }
    
    metadata = generator._validate_output(partial_output)
    assert metadata["status"] == "partial_success"
    assert metadata["error_rate"] == 0.14  # 1/7 sections failed
    assert "course_information" in metadata["error_sections"]
    
    # Test complete failure
    failed_output = {
        "course_information": {"error": "Failed"},
        "course_description_objectives": {"error": "Failed"},
        "course_content": {"error": "Failed"},
        "policies_procedures": {"error": "Failed"},
        "assessment_grading_criteria": {"error": "Failed"},
        "learning_resources": {"error": "Failed"},
        "course_schedule": {"error": "Failed"}
    }
    
    with pytest.raises(OutputValidationError) as exc_info:
        generator._validate_output(failed_output) 
    assert str(exc_info.value) == "Failed to generate any section."
