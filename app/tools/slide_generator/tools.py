from pydantic import BaseModel, Field
from typing import List, Optional,Union, Any
import os
from app.services.logger import setup_logger
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)

    with open(absolute_file_path, 'r') as file:
        return file.read()
    
class SlideGenerator:
    def __init__(self, args=None, vectorstore_class=Chroma, prompt=None, embedding_model=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.5-flash"),
            "embedding_model": GoogleGenerativeAIEmbeddings(model='models/embedding-001'),
            "parser": JsonOutputParser(pydantic_object=SlidePresentation),
            "prompt": read_text_file("prompt/slide_generator_prompt.txt"),
            "vectorstore_class": Chroma
        }

        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        self.embedding_model = embedding_model or default_config["embedding_model"]

        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.vectorstore, self.retriever, self.runner = None, None, None
        self.args = args
        self.verbose = verbose

        if vectorstore_class is None: raise ValueError("Vectorstore must be provided")
       

        # # Return the chain
        # prompt = PromptTemplate(
        #     template=self.prompt,
        #     input_variables=["attribute_collection"],
        #     partial_variables={"format_instructions": self.parser.get_format_instructions()}
        # )

        # if self.runner is None:
        #     logger.info(f"Creating vectorstore from {len(documents)} documents") if self.verbose else None
        #     self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)
        #     logger.info(f"Vectorstore created") if self.verbose else None

        #     self.retriever = self.vectorstore.as_retriever()
        #     logger.info(f"Retriever created successfully") if self.verbose else None

        #     self.runner = RunnableParallel(
        #         {"context": self.retriever,
        #         "attribute_collection": RunnablePassthrough()
        #         }
        #     )

        # chain = self.runner | prompt | self.model | self.parser

        # logger.info(f"Chain compilation complete")

        # return chain
    def validate_slides_content(self,response, topic, instructional_level):
        """Validates that slide content matches the requested topic and level."""
        # Check if topic keywords appear in slides
        topic_keywords = set(topic.lower().split())
        
        # Track coverage metrics
        topic_coverage = 0
        garbage_coverage =0
        template_requirements_met = False
        slides = response["slides"]
        whole_text_generated = ""

        for slide in slides:
            # Check topic relevance
            if slide["template"] == "twoColumn":
                 # Check template requirements
                template_requirements_met = True
                slide_text = (slide['title']+ " " + (slide["content"]["leftColumn"] or "") + " " + (slide["content"]["rightColumn"]or "")).lower()
                whole_text_generated += slide_text
               
                
            elif slide["template"] == "titleAndBullets":      
                contents = slide["content"]          
                slide_text = (slide['title'] + " " + " ".join(content for content  in contents)).lower()
                whole_text_generated += slide_text

            elif slide["template"] == "sectionHeader":      
                slide_text = (slide['title'] + " " + (slide["content"]["title"] or "")+ " " +(slide["content"]["subtitle"] if slide["content"]["subtitle"] else " " )).lower()
                whole_text_generated += slide_text
            else:
                slide_text = (slide['title'] + " " + (slide["content"] or "")).lower()
                whole_text_generated += slide_text
             #check if there are any * or /n in the slides
            if "*" in whole_text_generated or "\n" in whole_text_generated:
                garbage_coverage += 1
            matches = sum(1 for keyword in topic_keywords if keyword in slide_text)
            if matches > 0:
                topic_coverage += 1
          
              
        
        # Calculate coverage percentage
        coverage_percentage = (topic_coverage / len(response["slides"])) * 100
        garbage_coverage_percentage = (garbage_coverage / len(response["slides"])) * 100
        return {
            "topic_coverage": coverage_percentage,
            "template_requirements_met": template_requirements_met,
            "garbage_coverage_percentage": garbage_coverage_percentage,
            "valid": coverage_percentage > 70 and template_requirements_met and garbage_coverage_percentage ==0
        }

    def compile_with_context(self):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["instructional_level", "topic", "slides_titles"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.model | self.parser

        logger.info(f"Chain compilation complete")

        return chain

    def generate_slides(self):
        logger.info(f"Creating the Outlines for the Presentation") 

        chain = self.compile_with_context()
        

        input_parameters = {
            "instructional_level": self.args.instructional_level,
            "topic": self.args.topic,
            "slides_titles": self.args.slides_titles,
            "lang": self.args.lang
        }
        logger.info(f"Input parameters: {input_parameters}")

        response = chain.invoke(input_parameters)

        logger.info(f"Generated response: {response}")
         # Add validation metrics
        validation_results = self.validate_slides_content(response=response, topic=self.args.topic,instructional_level= self.args.instructional_level)
        logger.info(f"Response validation: {validation_results}")
        
        if not validation_results["valid"]:
            logger.warning(f"Generated content may not fully match the requested topic")
        return response

class Slide(BaseModel):
    title: str = Field(..., description="The title of the slide")
    template: str = Field(..., description="The slide template type: sectionHeader, titleAndBody, titleAndBullets, twoColumn")
    content: Optional[Union[str, list, dict, Any]] = Field(None, description="Content of the slide, can be string, list, dict, or any type")
    # subtitle: Optional[str] = Field(None, description="For sectionHeader slides, an optional subtitle")
    # body: Optional[str] = Field(None, description="For titleAndBody slides, a paragraph of text")
    # bullets: Optional[List[str]] = Field(None, description="For titleAndBullets slides, a list of bullet points")
    # leftColumn: Optional[str] = Field(None, description="For twoColumn slides, the left column content")
    # rightColumn: Optional[str] = Field(None, description="For twoColumn slides, the right column content")

class SlidePresentation(BaseModel):
    slides: List[Slide] = Field(..., description="The complete set of slides in the presentation")