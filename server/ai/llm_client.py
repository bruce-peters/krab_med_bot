import openai
import anthropic
import httpx
from typing import AsyncIterator, List, Dict, Optional
from server.config import settings
import logging
import tiktoken

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.ai_provider
        self.model = settings.llm_model
        
        if self.provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        elif self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif self.provider == "ollama":
            self.base_url = settings.ollama_base_url
            self.http_client = httpx.AsyncClient(timeout=30.0)
        elif self.provider == "mock":
            logger.info("Using mock LLM provider for testing")
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate AI response from conversation history"""
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        try:
            if self.provider == "openai":
                return await self._generate_openai(messages, temperature, max_tokens)
            elif self.provider == "anthropic":
                return await self._generate_anthropic(messages, temperature, max_tokens)
            elif self.provider == "ollama":
                return await self._generate_ollama(messages, temperature, max_tokens)
            elif self.provider == "mock":
                return await self._generate_mock(messages)
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, I'm having trouble responding right now. How can I help you?"

    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using OpenAI API"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using Anthropic Claude API"""
        # Anthropic requires system message separate
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        
        response = await self.client.messages.create(
            model=self.model,
            system=system_msg,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text

    async def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using local Ollama"""
        response = await self.http_client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def _generate_mock(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response for testing"""
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        ).lower()
        
        # Simple pattern matching for testing
        if "pain" in last_user_msg or "hurt" in last_user_msg:
            return "I'm sorry to hear you're in pain. On a scale of 1 to 10, how would you rate your pain level?"
        elif "dizzy" in last_user_msg or "dizziness" in last_user_msg:
            return "Dizziness can be concerning. How long have you been feeling dizzy? Is it constant or does it come and go?"
        elif "sleep" in last_user_msg:
            return "Sleep is so important for healing. Did you have trouble falling asleep, or did you wake up during the night?"
        elif "good" in last_user_msg or "fine" in last_user_msg or "okay" in last_user_msg:
            return "That's wonderful to hear! I'm glad you're feeling well. Is there anything else you'd like to share about how you're doing?"
        else:
            return "Thank you for sharing that with me. How are you feeling today? Are you experiencing any discomfort?"

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """Stream AI response token by token"""
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        try:
            if self.provider == "openai":
                async for chunk in await self._stream_openai(messages, temperature, max_tokens):
                    yield chunk
            elif self.provider == "anthropic":
                async for chunk in await self._stream_anthropic(messages, temperature, max_tokens):
                    yield chunk
            elif self.provider == "ollama":
                async for chunk in await self._stream_ollama(messages, temperature, max_tokens):
                    yield chunk
            elif self.provider == "mock":
                # Stream mock response word by word
                mock_response = await self._generate_mock(messages)
                for word in mock_response.split():
                    yield word + " "
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            yield "I apologize, I'm having trouble responding right now."

    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream response from OpenAI"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream response from Anthropic"""
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        
        async with self.client.messages.stream(
            model=self.model,
            system=system_msg,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream response from Ollama"""
        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text for cost estimation"""
        try:
            if self.provider == "openai":
                return len(self.tokenizer.encode(text))
            elif self.provider == "anthropic":
                # Approximation: ~4 chars per token
                return len(text) // 4
            elif self.provider == "ollama":
                # Approximation for local models
                return len(text) // 4
            else:
                return len(text.split())  # Word count approximation
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split())

    async def estimate_cost(self, messages: List[Dict[str, str]], response_tokens: int = 500) -> float:
        """Estimate API cost for request"""
        if self.provider not in ["openai", "anthropic"]:
            return 0.0  # Free for local models
        
        total_tokens = sum(await self.count_tokens(m["content"]) for m in messages)
        
        # Pricing as of Jan 2024 (update as needed)
        if self.provider == "openai":
            if "gpt-4" in self.model:
                input_cost = total_tokens * 0.00003  # $0.03 per 1K tokens
                output_cost = response_tokens * 0.00006  # $0.06 per 1K tokens
            else:  # GPT-3.5
                input_cost = total_tokens * 0.0000015  # $0.0015 per 1K tokens
                output_cost = response_tokens * 0.000002  # $0.002 per 1K tokens
        elif self.provider == "anthropic":
            if "opus" in self.model:
                input_cost = total_tokens * 0.000015  # $15 per 1M tokens
                output_cost = response_tokens * 0.000075  # $75 per 1M tokens
            else:  # Sonnet
                input_cost = total_tokens * 0.000003  # $3 per 1M tokens
                output_cost = response_tokens * 0.000015  # $15 per 1M tokens
        
        return input_cost + output_cost

llm_client = LLMClient(provider=settings.ai_provider)
