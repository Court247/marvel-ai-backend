import pytest
from app.tools.presentation_generator.slide_generator.core import executor
from app.tools.presentation_generator.slide_generator.tools import SlideGenerator, Slide,SlidePresentation


def test_slide_generator_executor_normal_operation(mocker):
    """Test the executor function with valid inputs."""
     # Mock the SlideGenerator to return a predefined response
    mock_slideGenerator = mocker.Mock(spec=SlideGenerator)
    mock_slideGenerator.generate_slides.return_value = {
        "slides": [
            {
                "title": "Introduction to Python",
                "template": "titleAndBullets",
                "content": ["Python is a programming language"]
            },
            {
                "title": "Basic Syntax",
                "template": "titleBody",
                "content": "Python syntax is simple"
            }
        ]
    }
    
    slides_titles = [
        "Introduction to Python",
        "Basic Syntax",
        "Control Structures"
    ]
    
    result = executor(
        slides_titles=slides_titles,
        topic="Python Programming",
        instructional_level="Beginner",
        lang="en"
    )
    print(result)
    assert result is not None
    # assert hasattr(result, 'slides')
    # assert len(result.slides) > 0
# @patch("app.tools.slide_generator.tools.GoogleGenerativeAIEmbeddings", autospec=True)
# @patch("app.tools.slide_generator.tools.GoogleGenerativeAI", autospec=True)
# @patch("app.tools.slide_generator.tools.JsonOutputParser", autospec=True)
# def test_slide_generator_executor_normal_operation(mock_parser, mock_gemini, mock_embeddings):
#     """Test the executor function with valid inputs."""
    
#     # Mock return values to prevent real API calls
#     mock_embeddings.return_value = MagicMock()
#     mock_gemini.return_value = MagicMock()
     
#     # Mock JsonOutputParser to return a valid response
#     mock_parser_instance = MagicMock()
#     mock_parser_instance.parse.return_value = {  
#         "slides": ["Introduction to Python", "Basic Syntax", "Control Structures"]
#     }
#     mock_parser.return_value = mock_parser_instance
#     slides_titles = [
#         "Introduction to Python",
#         "Basic Syntax",
#         "Control Structures"
#     ]
    
#     result = executor(
#         slides_titles=slides_titles,
#         topic="Python Programming",
#         instructional_level="Beginner",
#         lang="en"
#     )

#     print(result)

#     # Ensure the result is in the expected format
#     assert isinstance(result, dict), f"Expected dict, but got {type(result)}"
#     assert "slides" in result, "Key 'slides' not found in response"
#     assert isinstance(result["slides"], list), "Expected 'slides' to be a list"
# def test_slide_generator_executor_missing_inputs():
#     """Test the executor function with missing required inputs."""
#     with pytest.raises(ValueError):
#         executor(
#             slides_titles=[],
#             topic="",
#             instructional_level="",
#             lang="en"
#         )

# def test_slide_generator_init():
#     """Test initialization of SlideGenerator."""
#     args = SlideGeneratorInput(
#         slides_titles=["Intro", "Details"],
#         topic="Data Science",
#         instructional_level="Intermediate",
#         lang="en"
#     )
    
#     generator = SlideGenerator(args=args)
    
#     assert generator.args == args
#     assert generator.model is not None
#     assert generator.parser is not None
#     assert generator.embedding_model is not None

# def test_slide_generator_validate_slides_content():
#     """Test validate_slides_content method."""
#     generator = SlideGenerator(args=Mock())
    
#     mock_response = {
#         "slides": [
#             {
#                 "title": "Introduction to Machine Learning",
#                 "template": "twoColumn",
#                 "content": ["Machine learning is a subset of AI", "Algorithms learn from data"]
#             },
#             {
#                 "title": "Types of Machine Learning",
#                 "template": "titleAndBullets",
#                 "content": ["Supervised Learning", "Unsupervised Learning"]
#             }
#         ]
#     }
    
#     validation_result = generator.validate_slides_content(
#         response=mock_response, 
#         topic="Machine Learning", 
#         instructional_level="Beginner"
#     )
    
#     assert validation_result["valid"] is True
#     assert validation_result["topic_coverage"] > 0
#     assert validation_result["template_requirements_met"] is True

# def test_slide_generator_validate_slides_content_low_quality():
#     """Test validate_slides_content with low-quality content."""
#     generator = SlideGenerator(args=Mock())
    
#     mock_response = {
#         "slides": [
#             {
#                 "title": "Random Content",
#                 "template": "titleAndBody",
#                 "content": ["* Unrelated content\n", "Garbage text"]
#             }
#         ]
#     }
    
#     validation_result = generator.validate_slides_content(
#         response=mock_response, 
#         topic="Machine Learning", 
#         instructional_level="Beginner"
#     )
    
#     assert validation_result["valid"] is False
#     assert validation_result["topic_coverage"] == 0
#     assert validation_result["garbage_coverage_percentage"] > 0

# def test_slide_generator_compile_with_context():
#     """Test compilation of pipeline."""
#     args = SlideGeneratorInput(
#         slides_titles=["Intro", "Details"],
#         topic="Web Development",
#         instructional_level="Intermediate",
#         lang="en"
#     )
    
#     generator = SlideGenerator(args=args)
#     chain = generator.compile_with_context()
    
#     assert chain is not None

# def test_slide_generator_generate_slides():
#     """Test generate_slides method."""
#     args = SlideGeneratorInput(
#         slides_titles=["Cloud Overview", "Cloud Services", "Security"],
#         topic="Cloud Computing",
#         instructional_level="Advanced",
#         lang="en"
#     )
    
#     generator = SlideGenerator(args=args)
#     output = generator.generate_slides()
    
#     assert isinstance(output, SlidePresentation)
#     assert len(output.slides) > 0
    
#     for slide in output.slides:
#         assert isinstance(slide, Slide)
#         assert slide.title is not None
#         assert slide.template is not None
#         assert slide.content is not None

# def test_slide_model():
#     """Test the Slide Pydantic model."""
#     slide = Slide(
#         title="Introduction",
#         template="titleAndBullets",
#         content=["Key Point 1", "Key Point 2"]
#     )
    
#     assert slide.title == "Introduction"
#     assert slide.template == "titleAndBullets"
#     assert slide.content == ["Key Point 1", "Key Point 2"]

# def test_slide_presentation_model():
#     """Test the SlidePresentation Pydantic model."""
#     slides = [
#         Slide(title="Intro", template="titleAndBody", content="Overview"),
#         Slide(title="Details", template="twoColumn", content={"left": "Content1", "right": "Content2"})
#     ]
    
#     presentation = SlidePresentation(slides=slides)
    
#     assert len(presentation.slides) == 2
#     assert all(isinstance(slide, Slide) for slide in presentation.slides)