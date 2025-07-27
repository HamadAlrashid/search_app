import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
import numpy as np
from pydantic import Field

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        pipeline,
        TextIteratorStreamer # for streaming responses
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalLLM(BaseChatModel):
    """
    Local LLM wrapper for Hugging Face models that integrates with langchain.
    Supports popular models like Llama, Mistral, Qwen, etc.
    """
    
    # Declare fields as class attributes for Pydantic validation
    model_name: str = Field(default="microsoft/DialoGPT-medium", description="HuggingFace model name or local path")
    device: str = Field(default="auto", description="Device to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    
    tokenizer: Optional[Any] = Field(default=None, description="Loaded tokenizer instance")
    model: Optional[Any] = Field(default=None, description="Loaded model instance")
    pipeline: Optional[Any] = Field(default=None, description="Text generation pipeline")
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "auto",
        torch_dtype: str = "auto",
        quantization_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
            quantization_config: Configuration for quantization (4-bit, 8-bit)
            model_kwargs: Additional arguments for model loading
            tokenizer_kwargs: Additional arguments for tokenizer loading
            pipeline_kwargs: Additional arguments for pipeline creation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
        """
        super().__init__(**kwargs)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and sentence-transformers are required for local models. "
                "Install with: pip install transformers sentence-transformers torch"
            )
        
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Set up quantization if specified
        quantization = None
        if quantization_config:

            if quantization_config.get("4bit", False):
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    **quantization_config
                )
            elif quantization_config.get("8bit", False):
                quantization = BitsAndBytesConfig(
                    load_in_8bit=True,
                    **quantization_config
                )
        
        # Determine torch dtype
        if torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        
        # Set up device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local LLM: {model_name} on {device} with dtype {torch_dtype} and quantization {quantization}")
        
        try:
            # Load tokenizer
            tokenizer_kwargs = tokenizer_kwargs or {}
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                **tokenizer_kwargs
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = model_kwargs or {}
            model_kwargs.update({
                "torch_dtype": torch_dtype,
                "device_map": device if device != "cpu" else None,
                "quantization_config": quantization,
            })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            # Create text generation pipeline
            pipeline_kwargs = pipeline_kwargs or {}
            
            # Check if model is using accelerate (device_map is set)
            if hasattr(self.model, 'hf_device_map') or model_kwargs.get("device_map") is not None:
                # Model is using accelerate, don't specify device
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    **pipeline_kwargs
                )
            else:
                # Model is not using accelerate, specify device
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.model.device if hasattr(self.model, 'device') else device,
                    **pipeline_kwargs
                )
            
            logger.info(f"Successfully loaded local LLM: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load local LLM {model_name}: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "local_hf_llm"
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages for the model using appropriate chat template."""
        # Try to use the model's chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                # Convert to OpenAI format for chat template
                chat_messages = []
                for message in messages:
                    if isinstance(message, HumanMessage):
                        chat_messages.append({"role": "user", "content": message.content})
                    elif isinstance(message, AIMessage):
                        chat_messages.append({"role": "assistant", "content": message.content})
                    else:
                        chat_messages.append({"role": "user", "content": message.content})
                
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, falling back to simple format")
        
        # Fallback to simple format
        formatted_parts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_parts.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_parts.append(f"Assistant: {message.content}")
            else:
                formatted_parts.append(f"User: {message.content}")
        
        # Add assistant prompt at the end
        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """Generate response from messages."""
        
        # Format input
        formatted_input = self._format_messages(messages)
        
        # Set generation parameters with better stop sequences
        default_stop_sequences = ["Human:", "User:", "\nHuman:", "\nUser:", "<|im_end|>", "<|endoftext|>"]
        if stop:
            default_stop_sequences.extend(stop)
        
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "do_sample": kwargs.get("temperature", self.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False,  # Only return generated text, not input
        }
        
        try:
            # Generate response
            outputs = self.pipeline(
                formatted_input,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Clean up response - handle different pipeline return formats
            if generation_kwargs.get("return_full_text", True):
                # If return_full_text=True, remove the input prompt
                response = generated_text[len(formatted_input):].strip()
            else:
                # If return_full_text=False, use the generated text directly
                response = generated_text.strip()
            
            # Clean up response with stop sequences
            for stop_seq in default_stop_sequences:
                if stop_seq in response:
                    response = response.split(stop_seq)[0].strip()
                    break
            
            # Remove any remaining conversation artifacts
            # Remove lines that look like conversation starters
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('Human:', 'User:', 'Assistant:', 'AI:')):
                    cleaned_lines.append(line)
                elif line.startswith(('Human:', 'User:', 'Assistant:', 'AI:')):
                    # Stop at any conversation marker
                    break
            
            response = '\n'.join(cleaned_lines).strip()
            
            # Ensure we have some response
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try again."
            
            # Create AI message
            ai_message = AIMessage(content=response)
            generation = ChatGeneration(message=ai_message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            ai_message = AIMessage(content=f"Error generating response: {str(e)}")
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> Iterator[ChatGeneration]:
        """Stream response from messages."""
        # TODO: Implement streaming
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]


class LocalEmbeddings(Embeddings):
    """
    Local embedding model wrapper using sentence-transformers.
    Supports various embedding models like all-MiniLM, BGE, E5, etc.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        normalize_embeddings: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize local embedding model.
        
        Args:
            model_name: Sentence-transformers model name or local path
            device: Device to use ("auto", "cuda", "cpu")
            normalize_embeddings: Whether to normalize embeddings
            model_kwargs: Additional arguments for model loading
            encode_kwargs: Additional arguments for encoding
        """
        super().__init__(**kwargs)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and sentence-transformers are required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.encode_kwargs = encode_kwargs or {}
        
        # Set up device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local embedding model: {model_name} on {device}")
        
        try:
            model_kwargs = model_kwargs or {}
            self.model = SentenceTransformer(
                model_name,
                device=device,
                **model_kwargs
            )
            logger.info(f"Successfully loaded local embedding model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load local embedding model {model_name}: {e}")
            raise
   
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of texts to embed
        """
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                **self.encode_kwargs
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
        """
        try:
            embedding = self.model.encode(
                [text],
                normalize_embeddings=self.normalize_embeddings,
                **self.encode_kwargs
            )
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


# Popular model configurations
POPULAR_LOCAL_LLMS = {
    # Small models (good for CPU)
    "microsoft/DialoGPT-medium": {
        "description": "Conversational model, good for CPU",
        "size": "~350MB",
        "recommended_device": "cpu"
    },
    "distilgpt2": {
        "description": "Lightweight GPT-2 variant",
        "size": "~340MB", 
        "recommended_device": "cpu"
    },
    
    # Medium models (good for GPU with 8GB+)
    "microsoft/phi-2": {
        "description": "2.7B parameters, efficient",
        "size": "~5GB",
        "recommended_device": "cuda",
        
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "description": "1.1B parameter Llama model",
        "size": "~2GB",
        "recommended_device": "cuda"
    },
    
    # Large models (require GPU with 16GB+)
    "meta-llama/Llama-2-7b-chat-hf": {
        "description": "Llama 2 7B chat model",
        "size": "~13GB",
        "recommended_device": "cuda",
        "quantization": {"4bit": True}
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "description": "Mistral 7B instruction-tuned",
        "size": "~13GB", 
        "recommended_device": "cuda",
        "quantization": {"4bit": True}
    },
    "Qwen/Qwen1.5-7B-Chat": {
        "description": "Qwen 7B chat model",
        "size": "~13GB",
        "recommended_device": "cuda",
        "quantization": {"4bit": True}
    }
}

POPULAR_LOCAL_EMBEDDINGS = {
    # Small, fast models
    "all-MiniLM-L6-v2": {
        "description": "Fast and lightweight, 384 dimensions",
        "size": "~90MB",
        "dimensions": 384
    },
    "all-MiniLM-L12-v2": {
        "description": "Better quality than L6, 384 dimensions", 
        "size": "~130MB",
        "dimensions": 384
    },
    
    # High quality models
    "all-mpnet-base-v2": {
        "description": "High quality general purpose, 768 dimensions",
        "size": "~420MB",
        "dimensions": 768
    },
    "BAAI/bge-small-en-v1.5": {
        "description": "BGE small model, high quality",
        "size": "~130MB", 
        "dimensions": 384
    },
    "BAAI/bge-base-en-v1.5": {
        "description": "BGE base model, very high quality",
        "size": "~440MB",
        "dimensions": 768
    },
    
    # Qwen3 embedding models (Latest & Best Performance)
    "Qwen/Qwen3-Embedding-0.6B": {
        "description": "Qwen3 0.6B embedding model, multilingual, 100+ languages",
        "size": "~1.2GB",
        "dimensions": 1024
    },
    "Qwen/Qwen3-Embedding-4B": {
        "description": "Qwen3 4B embedding model, state-of-the-art performance",
        "size": "~8GB",
        "dimensions": 2560
    },
    "Qwen/Qwen3-Embedding-8B": {
        "description": "Qwen3 8B embedding model, #1 on MTEB multilingual leaderboard",
        "size": "~16GB",
        "dimensions": 4096
    },
    
    # Advanced embedding models  
    "BAAI/bge-large-en-v1.5": {
        "description": "BGE large model, best quality",
        "size": "~1.3GB",
        "dimensions": 1024
    },
    "BAAI/bge-m3": {
        "description": "BGE multilingual model, supports 100+ languages",
        "size": "~2.3GB",
        "dimensions": 1024
    },
    
    # Specialized high-performance models
    "jinaai/jina-embeddings-v2-base-en": {
        "description": "Jina embeddings v2, high quality for English",
        "size": "~550MB",
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v2-small-en": {
        "description": "Jina embeddings v2 small, efficient for English",
        "size": "~130MB",
        "dimensions": 512
    },
    "intfloat/e5-large-v2": {
        "description": "E5 large model, excellent performance",
        "size": "~1.3GB",
        "dimensions": 1024
    },
    "intfloat/e5-base-v2": {
        "description": "E5 base model, good balance of quality and speed",
        "size": "~440MB",
        "dimensions": 768
    },
    "intfloat/e5-small-v2": {
        "description": "E5 small model, fast and efficient",
        "size": "~130MB",
        "dimensions": 384
    },
    
    # Multilingual models
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "description": "Multilingual support, 384 dimensions",
        "size": "~470MB",
        "dimensions": 384
    },
    "intfloat/multilingual-e5-large": {
        "description": "Multilingual E5 large, supports 100+ languages",
        "size": "~2.2GB", 
        "dimensions": 1024
    },
    "intfloat/multilingual-e5-base": {
        "description": "Multilingual E5 base, good quality for many languages",
        "size": "~1.1GB",
        "dimensions": 768
    },
    
    # Jina AI multilingual models
    "jinaai/jina-embeddings-v2-base-zh": {
        "description": "Jina v2 Chinese/English bilingual model",
        "size": "~550MB",
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v2-base-de": {
        "description": "Jina v2 German/English bilingual model", 
        "size": "~550MB",
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v2-base-es": {
        "description": "Jina v2 Spanish/English bilingual model",
        "size": "~550MB", 
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v3": {
        "description": "Jina v3 multilingual model, supports 100+ languages",
        "size": "~570MB",
        "dimensions": 1024
    },
    "jinaai/jina-clip-v1": {
        "description": "Jina CLIP model for text and image embeddings",
        "size": "~600MB",
        "dimensions": 768
    },
    
    # Jina AI multilingual models
    "jinaai/jina-embeddings-v2-base-zh": {
        "description": "Jina v2 Chinese/English bilingual model",
        "size": "~550MB",
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v2-base-de": {
        "description": "Jina v2 German/English bilingual model", 
        "size": "~550MB",
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v2-base-es": {
        "description": "Jina v2 Spanish/English bilingual model",
        "size": "~550MB", 
        "dimensions": 768
    },
    "jinaai/jina-embeddings-v3": {
        "description": "Jina v3 multilingual model, supports 100+ languages",
        "size": "~570MB",
        "dimensions": 1024
    },
    "jinaai/jina-clip-v1": {
        "description": "Jina CLIP model for text and image embeddings",
        "size": "~600MB",
        "dimensions": 768
    }
}


def get_local_llm(
    model_name: str,
    device: str = "auto",
    use_quantization: bool = True,
    **kwargs
) -> LocalLLM:
    """
    Helper function to create a local LLM with recommended settings.
    
    Args:
        model_name: Model name or key from POPULAR_LOCAL_LLMS
        device: Device to use
        use_quantization: Whether to use quantization for large models
        **kwargs: Additional arguments for LocalLLM
    
    Returns:
        Configured LocalLLM instance
    """
    # Get model config if available
    model_config = POPULAR_LOCAL_LLMS.get(model_name, {})
    
    # Set up quantization
    quantization_config = None
    if use_quantization and model_config.get("quantization"):
        quantization_config = model_config["quantization"]
    
    # Override device if recommended
    if device == "auto" and "recommended_device" in model_config:
        device = model_config["recommended_device"]
    
    return LocalLLM(
        model_name=model_name,
        device=device,
        quantization_config=quantization_config,
        **kwargs
    )


def get_local_embeddings(
    model_name: str,
    device: str = "auto",
    **kwargs
) -> LocalEmbeddings:
    """
    Helper function to create local embeddings with recommended settings.
    
    Args:
        model_name: Model name or key from POPULAR_LOCAL_EMBEDDINGS  
        device: Device to use
        **kwargs: Additional arguments for LocalEmbeddings
    
    Returns:
        Configured LocalEmbeddings instance
    """
    return LocalEmbeddings(
        model_name=model_name,
        device=device,
        **kwargs
    )


