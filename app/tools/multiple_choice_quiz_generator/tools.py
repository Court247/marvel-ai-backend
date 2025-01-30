from typing import List, Dict
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.services.logger import setup_logger

relative_path = "tools/multiple_choice_quiz_generator"

logger = setup_logger(__name__)

def transform_json_dict(input_data: dict) -> dict:
    # Validate and parse the input data to ensure it matches the QuizQuestion schema
    quiz_question = QuizQuestion(**input_data)

    # Transform the choices list into a dictionary
    transformed_choices = {choice.key: choice.value for choice in quiz_question.choices}

    # Create the transformed structure
    transformed_data = {
        "question": quiz_question.question,
        "choices": transformed_choices,
        "answer": quiz_question.answer,
        "explanation": quiz_question.explanation
    }

    return transformed_data

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class QuizBuilder:
    """
    A class for building and generating multiple-choice quiz questions based on provided documents.
    
    This class uses LangChain components to process documents, generate embeddings, and create
    quiz questions using a language model.
    """
    
    def __init__(self, topic, lang='en', vectorstore_class=Chroma, prompt=None, embedding_model=None, model=None, parser=None, verbose=False):
        """
        Initialize the QuizBuilder with configuration and models.
        
        Args:
            topic (str): The main topic for quiz questions
            lang (str): Language code for question generation (default: 'en')
            vectorstore_class: Vector store class for document embeddings (default: Chroma)
            prompt (str): Custom prompt template (optional)
            embedding_model: Model for generating embeddings (optional)
            model: Language model for question generation (optional)
            parser: Output parser for JSON responses (optional)
            verbose (bool): Enable detailed logging (default: False)
        """
        # Default configuration for models and components
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.0-pro"),
            "embedding_model": GoogleGenerativeAIEmbeddings(model='models/embedding-001'),
            "parser": JsonOutputParser(pydantic_object=QuizQuestion),
            "prompt": read_text_file("prompt/multiple_choice_quiz_generator_prompt.txt"),
            "vectorstore_class": Chroma
        }
        
        # Initialize components with provided values or defaults
        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        
        # Initialize vector store related attributes
        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.vectorstore, self.retriever, self.runner = None, None, None
        
        # Set basic configuration
        self.topic = topic
        self.lang = lang
        self.verbose = verbose
        
        # Validate essential parameters
        if vectorstore_class is None: raise ValueError("Vectorstore must be provided")
        if topic is None: raise ValueError("Topic must be provided")
    
    def compile(self, documents: List[Document], num_questions: int = 1):
        """
        Compile the question generation chain using the provided documents.
        
        Args:
            documents (List[Document]): List of documents to use as context
            num_questions (int): Number of questions to generate per prompt (default: 1)
            
        Returns:
            Chain: A compiled LangChain chain for question generation
        """
        # Initialize prompt template with all required variables
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["attribute_collection", "context", "num_questions"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # Create vector store and retriever if not already initialized
        if self.runner is None:
            # Log vector store creation if verbose mode is enabled
            logger.info(f"Creating vectorstore from {len(documents)} documents") if self.verbose else None
            self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)
            logger.info(f"Vectorstore created") if self.verbose else None

            # Initialize retriever from vector store
            self.retriever = self.vectorstore.as_retriever()
            logger.info(f"Retriever created successfully") if self.verbose else None

            # Set up parallel running configuration with all required components
            self.runner = RunnableParallel(
                {
                    "context": self.retriever,  # Retrieves relevant context from documents
                    "attribute_collection": RunnablePassthrough(),  # Passes through the topic and language
                    "num_questions": lambda _: str(num_questions)  # Converts num_questions to string for template
                }
            )
        
        # Compile the full chain: runner -> prompt -> model -> parser
        chain = self.runner | prompt | self.model | self.parser
        
        logger.info(f"Chain compilation complete") if self.verbose else None
        
        return chain

    def create_questions(self, documents: List[Document], num_questions: int = 5) -> List[Dict]:
        """
        Generate multiple-choice quiz questions based on the provided documents.
        
        Args:
            documents (List[Document]): List of documents to use as context
            num_questions (int): Number of questions to generate (default: 5)
            
        Returns:
            List[Dict]: List of generated quiz questions with choices and answers
        """
        logger.info(f"Creating {num_questions} questions") if self.verbose else None
        
        # Validate number of questions
        if num_questions > 10:
            return {"message": "error", "data": "Number of questions cannot exceed 10"}
        
        # Compile the chain with single question generation
        chain = self.compile(documents, num_questions=1)
        
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 5  # Set maximum attempts to avoid infinite loops
        
        # Generate questions until target number is reached or max attempts exceeded
        while len(generated_questions) < num_questions and attempts < max_attempts:
            # Generate a single question
            response = chain.invoke(f"Topic: {self.topic}, Lang: {self.lang}")
            if self.verbose:
                logger.info(f"Generated response attempt {attempts + 1}: {response}")

            # Transform and validate the response
            response = transform_json_dict(response)
            if self.validate_response(response):
                # Format choices and add valid question to results
                response["choices"] = self.format_choices(response["choices"])
                generated_questions.append(response)
                if self.verbose:
                    logger.info(f"Valid question added: {response}")
                    logger.info(f"Total generated questions: {len(generated_questions)}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            
            attempts += 1

        # Log warning if fewer questions were generated than requested
        if len(generated_questions) < num_questions:
            logger.warning(f"Only generated {len(generated_questions)} out of {num_questions} requested questions")
        
        # Clean up vector store
        if self.verbose: logger.info(f"Deleting vectorstore")
        self.vectorstore.delete_collection()
        
        # Return requested number of questions (or fewer if not enough were generated)
        return generated_questions[:num_questions]
    
    #new function to validate the response
    def validate_response(self, response: dict) -> bool:
        """
        Validates the structure and content of a quiz question response.
        
        Args:
            response (dict): The quiz question response to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if all required fields are present
            required_fields = ['question', 'choices', 'answer', 'explanation']
            if not all(field in response for field in required_fields):
                logger.warning("Missing required fields") if self.verbose else None
                return False
            
            # Validate choices
            if not isinstance(response['choices'], dict):
                logger.warning("Choices must be a dictionary") if self.verbose else None
                return False
            
            # Validate that there are exactly 4 choices
            if len(response['choices']) != 4:
                logger.warning("Must have exactly 4 choices") if self.verbose else None
                return False
            
            # Validate choice keys are A, B, C, D
            valid_keys = set(['A', 'B', 'C', 'D'])
            if set(response['choices'].keys()) != valid_keys:
                logger.warning("Choice keys must be A, B, C, D") if self.verbose else None
                return False
            
            # Validate answer is one of the valid keys
            if response['answer'] not in valid_keys:
                logger.warning("Answer must be one of: A, B, C, D") if self.verbose else None
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}") if self.verbose else None
            return False

    def format_choices(self, choices: dict) -> dict:
        """
        Formats the choices to ensure they are in the correct order and structure.
        
        Args:
            choices (dict): Dictionary of choices with keys A, B, C, D
            
        Returns:
            dict: Formatted choices in correct order
        """
        ordered_keys = ['A', 'B', 'C', 'D']
        return {key: choices[key] for key in ordered_keys if key in choices}


class QuestionChoice(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, or D.")
    value: str = Field(description="The text content of the choice")
class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoice] = Field(description="A list of choices for the question, each with a key and a value")
    answer: str = Field(description="The key of the correct answer from the choices list")
    explanation: str = Field(description="An explanation of why the answer is correct")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "What is the capital of France?",
                "choices": [
                    {"key": "A", "value": "Berlin"},
                    {"key": "B", "value": "Madrid"},
                    {"key": "C", "value": "Paris"},
                    {"key": "D", "value": "Rome"}
                ],
                "answer": "C",
                "explanation": "Paris is the capital of France."
              }
          """
        }

      }

