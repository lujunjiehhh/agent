# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import os
import re
import time
from typing import Callable, List, Literal, Optional, Union

import openai
from langchain_core.embeddings import Embeddings
from openai import OpenAI, AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from pickledb import PickleDB

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile('(?<!\\\\)\'').sub('\"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        wait_time = int(matches.group(1)) + 3  # NOTE: wait 3 more seconds here.
        return wait_time
    except Exception as e:
        return None


class OpenAIClient(BaseLLMClient):
    NAME = "OpenAIClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        """LLM Communication Client for OpenAI endpoints.

        Args:
            location (str): the file location of the LLM client communication cache. No cache would be created if set to
                None. Defaults to None.
            auto_dump (bool): automatically save the Client's communication cache or not. Defaults to True.
            logger (Logger): client logger. Defaults to None.
            max_attempt (int): Maximum attempt time for LLM requesting. Request would be skipped if max_attempt reached.
                Defaults to 5.
            exponential_backoff_factor (int): Set to enable exponential backoff retry manner. Every time the wait time
                would be `exponential_backoff_factor ^ num_attempt`. Set to None to disable and use the `unit_wait_time`
                manner. Defaults to None.
            unit_wait_time (int): `unit_wait_time` would be used only if the exponential backoff mode is disabled. Every
                time the wait time would be `unit_wait_time * num_attempt`, with seconds (s) as the time unit. Defaults
                to 60.
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("OPENAI_API_KEY", None) is None:
            raise ValueError("OpenAI API key must be provided either in client_config or as environment variable OPENAI_API_KEY")

        self._client = OpenAI(**client_configs)
        self._async_client = AsyncOpenAI(**client_configs)
        self._model = kwargs.get("model_name", "gpt-4-turbo-preview")

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        response: ChatCompletion = None
        num_attempt: int = 0
        
        # 使用配置中的model_name或默认值
        model = llm_config.pop("model_name", self._model)
        
        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **llm_config
                )
                break

            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning(f"  Retrying...")

            except openai.BadRequestError as e:
                self.warning(f"  Failed due to Exception: {e}")
                self.warning(f"  Skip this request...")
                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    async def _async_get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        response: ChatCompletion = None
        num_attempt: int = 0
        
        # 使用配置中的model_name或默认值
        model = llm_config.pop("model_name", self._model)
        
        while num_attempt < self._max_attempt:
            try:
                response = await self._async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **llm_config
                )
                break

            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                wait_time = parse_wait_time_from_error(e)
                await self._async_wait(num_attempt, wait_time=wait_time)
                self.warning(f"  Retrying...")

            except openai.BadRequestError as e:
                self.warning(f"  Failed due to Exception: {e}")
                self.warning(f"  Skip this request...")
                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                await self._async_wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        try:
            content = response.choices[0].message.content
            if content is None:
                finish_reason = response.choices[0].finish_reason
                warning_message = f"Non-Content returned due to {finish_reason}"

                if "content_filter" in finish_reason:
                    for reason, res_dict in response.choices[0].content_filter_results.items():
                        if res_dict["filtered"] is True or res_dict["severity"] != "safe":
                            warning_message += f", '{reason}': {res_dict}"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except Exception as e:
            self.warning(f"Try to get content from response but get exception:\n  {e}")
            self.debug(
                f"  Response: {response}\n"
                f"  Last message: {messages}"
            )
            content = ""

        return content

    def close(self):
        super().close()
        self._client.close()

    async def aclose(self):
        """异步关闭连接"""
        await super().aclose()
        await self._async_client.close()


class OpenAIEmbedding(Embeddings):
    def __init__(self, **kwargs) -> None:
        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("OPENAI_API_KEY", None) is None:
            raise ValueError("OpenAI API key must be provided either in client_config or as environment variable OPENAI_API_KEY")

        self._client = OpenAI(**client_configs)
        self._async_client = AsyncOpenAI(**client_configs)
        self._model = kwargs.get("model_name", "text-embedding-3-small")

        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        auto_dump = cache_config.get("auto_dump", True)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
        else:
            self._cache = None

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        if self._cache is None:
            return

        self._cache.set(query, embedding)
        return

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        if self._cache is None:
            return False

        return self._cache.get(query)

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        while True:
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts
                )
                break

            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    time.sleep(expected_wait)
                else:
                    print(f"Embedding failed due to RateLimitError, but failed parsing expected waiting time, wait for 30 seconds")
                    time.sleep(30)

            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                exit(0)

        return response

    async def _async_get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        while True:
            try:
                response = await self._async_client.embeddings.create(
                    model=self._model,
                    input=texts
                )
                break

            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    await asyncio.sleep(expected_wait)
                else:
                    print(f"Embedding failed due to RateLimitError, but failed parsing expected waiting time, wait for 30 seconds")
                    await asyncio.sleep(30)

            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                exit(0)

        return response

    def embed_documents(self, texts: List[str], batch_call: bool=False) -> List[List[float]]:
        if batch_call is True:
            response = self._get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    async def aembed_documents(self, texts: List[str], batch_call: bool=False) -> List[List[float]]:
        if batch_call is True:
            response = await self._async_get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = await asyncio.gather(*[self.aembed_query(text) for text in texts])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        embedding = self._get_cache(text)
        if embedding is False:
            response = await self._async_get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding 