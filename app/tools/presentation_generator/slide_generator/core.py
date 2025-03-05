
from app.api.error_utilities import LoaderError, ToolExecutorError
from typing import List
from app.services.schemas import SlideGeneratorInput
from app.tools.presentation_generator.slide_generator.tools import SlideGenerator
from app.services.logger import setup_logger
logger = setup_logger()

@pytest.fixture
def mock_slide_data():
    return {
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

def executor(
             slides_titles: List[str],
             topic: str,
             instructional_level: str,
             lang: str, 
             verbose=False):
    try: 
        if (not (slides_titles and topic and instructional_level)):
            logger.info(f"Missing required inputs.")
         
        if(slides_titles and topic and instructional_level):
            logger.info(f"Generating slide outlines. from {topic} for {instructional_level} level")
        
       
        slide_generator_args = SlideGeneratorInput(
            slides_titles=slides_titles,
            instructional_level=instructional_level, 
            topic=topic,
            lang=lang
        )
        output = SlideGenerator(args=slide_generator_args, verbose=verbose).generate_slides()
        logger.info(f"Slides generated successfully")
    except LoaderError as e:
        error_message = e
        logger.error(f"Error in Slide Generator Pipeline -> {error_message}")
        raise ToolExecutorError(error_message)
    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    return output