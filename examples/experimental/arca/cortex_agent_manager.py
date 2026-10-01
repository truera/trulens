"""
Snowflake Cortex Agent Manager
Handles creation, configuration, and interaction with Cortex Agents via REST API.

Authentication: Uses Programmatic Access Token (PAT)
"""

from dataclasses import dataclass
from dataclasses import field
import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class AgentInstructions:
    """Agent instruction parameters - these are the main tunable knobs."""

    response: str = ""  # How to format responses
    orchestration: str = ""  # How to choose and use tools
    system: str = ""  # Overall agent behavior and personality
    sample_questions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AgentProfile:
    """Visual profile for the agent."""

    display_name: str
    avatar: Optional[str] = None
    color: Optional[str] = None


@dataclass
class BudgetConfig:
    """Resource constraints for agent execution."""

    seconds: int = 30
    tokens: int = 16000


@dataclass
class ModelConfig:
    """Model configuration for agent."""

    orchestration: str = "claude-4-sonnet"  # Model for orchestration


@dataclass
class ToolSpec:
    """Specification for a tool the agent can use."""

    type: str  # 'cortex_analyst_text_to_sql', 'cortex_search', 'generic'
    name: str
    description: str  # Tunable - affects tool selection
    input_schema: Optional[Dict[str, Any]] = None


@dataclass
class ToolResource:
    """Configuration for tool execution."""

    type: str  # 'function', 'cortex_analyst', 'cortex_search'
    config: Dict[str, Any] = field(default_factory=dict)


class CortexAgentManager:
    """
    Manager for Snowflake Cortex Agents via REST API.

    Auth Requirements:
    1. PAT (Programmatic Access Token) from Snowflake
    2. USAGE privileges on database, schema, and resources
    """

    def __init__(
        self,
        account_url: str,
        auth_token: str,
        database: str,
        schema: str,
    ):
        """
        Initialize the Cortex Agent Manager.

        Args:
            account_url: Snowflake account URL (e.g., https://myaccount.snowflakecomputing.com)
            auth_token: PAT (Programmatic Access Token)
            database: Database name where agent will be created
            schema: Schema name where agent will be created
        """
        self.account_url = account_url.rstrip("/")
        self.auth_token = auth_token
        self.database = database
        self.schema = schema
        self.base_api_url = f"{self.account_url}/api/v2"

        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_agent(
        self,
        name: str,
        instructions: AgentInstructions,
        tools: List[Dict[str, Any]],
        tool_resources: Dict[str, Dict[str, Any]],
        profile: Optional[AgentProfile] = None,
        models: Optional[ModelConfig] = None,
        budget: Optional[BudgetConfig] = None,
        comment: str = "",
        create_mode: str = "errorIfExists",
    ) -> Dict[str, Any]:
        """
        Create a new Cortex Agent.

        Args:
            name: Agent name
            instructions: Agent instructions (main tunable parameters)
            tools: List of tool specifications
            tool_resources: Tool execution configurations
            profile: Visual profile for the agent
            models: Model configuration
            budget: Resource budget constraints
            comment: Optional comment
            create_mode: 'errorIfExists' or 'orReplace' or 'ifNotExists'

        Returns:
            Response from the API
        """
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents"

        # Build request body
        body: Dict[str, Any] = {
            "name": name,
            "comment": comment,
            "instructions": {
                "response": instructions.response,
                "orchestration": instructions.orchestration,
                "system": instructions.system,
            },
            "tools": tools,
            "tool_resources": tool_resources,
        }

        if instructions.sample_questions:
            body["instructions"]["sample_questions"] = (
                instructions.sample_questions
            )

        if profile:
            body["profile"] = {
                "display_name": profile.display_name,
            }
            if profile.avatar:
                body["profile"]["avatar"] = profile.avatar
            if profile.color:
                body["profile"]["color"] = profile.color

        if models:
            body["models"] = {
                "orchestration": models.orchestration,
            }

        if budget:
            body["orchestration"] = {
                "budget": {
                    "seconds": budget.seconds,
                    "tokens": budget.tokens,
                }
            }

        params = {"createMode": create_mode}

        logger.info(f"Creating agent: {name}")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request params: {params}")
        logger.debug(f"Request body: {json.dumps(body, indent=2)}")

        response = requests.post(
            url, json=body, headers=self.headers, params=params
        )

        if not response.ok:
            logger.error(
                f"Error creating agent. Status: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            try:
                error_detail = response.json()
                logger.error(
                    f"Error details: {json.dumps(error_detail, indent=2)}"
                )
            except Exception:
                pass

        response.raise_for_status()

        return response.json()

    def update_agent(
        self,
        name: str,
        instructions: Optional[AgentInstructions] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Dict[str, Any]]] = None,
        models: Optional[ModelConfig] = None,
        budget: Optional[BudgetConfig] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing Cortex Agent.

        Args:
            name: Agent name
            instructions: Updated instructions (only provided fields will be updated)
            tools: Updated tool specifications
            tool_resources: Updated tool configurations
            models: Updated model configuration
            budget: Updated budget constraints
            comment: Updated comment

        Returns:
            Response from the API
        """
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents/{name}"

        body: Dict[str, Any] = {}

        if comment is not None:
            body["comment"] = comment

        if instructions:
            body["instructions"] = {}
            if instructions.response:
                body["instructions"]["response"] = instructions.response
            if instructions.orchestration:
                body["instructions"]["orchestration"] = (
                    instructions.orchestration
                )
            if instructions.system:
                body["instructions"]["system"] = instructions.system
            if instructions.sample_questions:
                body["instructions"]["sample_questions"] = (
                    instructions.sample_questions
                )

        if tools is not None:
            body["tools"] = tools

        if tool_resources is not None:
            body["tool_resources"] = tool_resources

        if models:
            body["models"] = {"orchestration": models.orchestration}

        if budget:
            body["orchestration"] = {
                "budget": {
                    "seconds": budget.seconds,
                    "tokens": budget.tokens,
                }
            }

        logger.info(f"Updating agent: {name}")
        response = requests.put(url, json=body, headers=self.headers)
        response.raise_for_status()

        # API may return empty response (204 No Content)
        if response.content:
            return response.json()
        return {"status": "success"}

    def describe_agent(self, name: str) -> Dict[str, Any]:
        """Get details about an agent."""
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents/{name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_agent(self, name: str) -> Dict[str, Any]:
        """Delete an agent."""
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents/{name}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents in the schema."""
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    # Thread and Message Management

    def create_thread(self, origin_application: str = "arca") -> str:
        """
        Create a new conversation thread using the global Threads API.

        Args:
            origin_application: Name of the application creating the thread (max 16 bytes)

        Returns:
            Thread UUID as string
        """
        url = f"{self.account_url}/api/v2/cortex/threads"
        body = {"origin_application": origin_application}

        logger.debug(
            f"Creating thread with origin_application: {origin_application}"
        )
        response = requests.post(url, headers=self.headers, json=body)

        if not response.ok:
            logger.error(
                f"Error creating thread. Status: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")

        response.raise_for_status()

        # API returns thread metadata object
        thread_data = response.json()
        # Extract thread_id from the response
        thread_id = str(thread_data.get("thread_id", thread_data))
        logger.debug(f"Created thread ID: {thread_id}")
        return thread_id

    def send_message(
        self,
        agent_name: str,
        thread_id: str,
        message: str,
        parent_message_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to the agent and get a response using the streaming :run endpoint.

        Uses: POST /api/v2/databases/{database}/schemas/{schema}/agents/{name}:run

        Args:
            agent_name: Name of the agent
            thread_id: Thread ID for the conversation
            message: User message
            parent_message_id: ID of parent message (0 for first message)
            metadata: Optional metadata

        Returns:
            Dict containing:
                - message: Agent response with role and content
                - tool_uses: List of tools used during processing
                - tool_results: Results from tool executions
        """
        url = f"{self.base_api_url}/databases/{self.database}/schemas/{self.schema}/agents/{agent_name}:run"

        # Build request body according to API spec
        request_body = {
            "thread_id": int(thread_id),
            "parent_message_id": parent_message_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message}],
                }
            ],
        }

        logger.debug(
            f"Sending message to agent {agent_name} in thread {thread_id}"
        )
        logger.debug(f"Request body: {json.dumps(request_body, indent=2)}")

        # Use streaming response
        response = requests.post(
            url, json=request_body, headers=self.headers, stream=True
        )

        if not response.ok:
            logger.error(f"Error running agent. Status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            response.raise_for_status()

        # Parse streaming server-sent events
        assistant_text = ""
        tool_uses = []
        tool_results = []
        event_type = None

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            logger.debug(f"SSE line: {line}")

            # Parse SSE format: "event: <type>\ndata: <json>"
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
                logger.debug(f"Event type: {event_type}")
            elif line.startswith("data:"):
                try:
                    data_str = line.split(":", 1)[1].strip()
                    data = json.loads(data_str)
                    logger.debug(f"Event data: {json.dumps(data, indent=2)}")

                    # Handle different event types
                    if event_type == "response.text.delta":
                        # API returns "text" field, not "delta"
                        text_chunk = data.get("text", "")
                        assistant_text += text_chunk
                        logger.debug(f"Added text chunk: {text_chunk}")
                    elif event_type == "response.tool_use":
                        tool_uses.append(data)
                    elif event_type == "response.tool_result":
                        tool_results.append(data)
                    elif event_type == "error":
                        logger.error(f"Agent error: {data}")
                        raise Exception(f"Agent error: {data}")

                except json.JSONDecodeError as e:
                    logger.debug(
                        f"Could not parse SSE data: {data_str}, error: {e}"
                    )
                    continue

        logger.info(f"Parsed assistant text: {assistant_text}")

        # Capture request ID from response headers
        request_id = response.headers.get("X-Snowflake-Request-Id")

        # Return formatted response
        return {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
            "tool_uses": tool_uses,
            "tool_results": tool_results,
            "request_id": request_id,
        }

    def get_thread_messages(
        self,
        thread_id: str,
        page_size: int = 20,
        last_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get messages in a thread using the global Threads API.

        Args:
            thread_id: Thread UUID
            page_size: Number of messages to return (default: 20, max: 100)
            last_message_id: ID of last message received, for pagination

        Returns:
            Thread metadata and messages
        """
        url = f"{self.account_url}/api/v2/cortex/threads/{thread_id}"
        params = {"page_size": page_size}
        if last_message_id:
            params["last_message_id"] = last_message_id

        response = requests.get(url, headers=self.headers, params=params)

        if not response.ok:
            logger.error(
                f"Error getting thread messages. Status: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")

        response.raise_for_status()
        return response.json()

    def list_threads(
        self, origin_application: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all threads using the global Threads API.

        Args:
            origin_application: Optional filter by application name

        Returns:
            List of thread metadata objects
        """
        url = f"{self.account_url}/api/v2/cortex/threads"
        params = {}
        if origin_application:
            params["origin_application"] = origin_application

        response = requests.get(url, headers=self.headers, params=params)

        if not response.ok:
            logger.error(
                f"Error listing threads. Status: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")

        response.raise_for_status()
        return response.json()

    def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        """
        Delete a thread using the global Threads API.

        Args:
            thread_id: Thread UUID

        Returns:
            Success response
        """
        url = f"{self.account_url}/api/v2/cortex/threads/{thread_id}"
        response = requests.delete(url, headers=self.headers)

        if not response.ok:
            logger.error(
                f"Error deleting thread. Status: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")

        response.raise_for_status()
        return response.json()


def build_generic_tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to build a generic tool specification."""
    return {
        "tool_spec": {
            "type": "generic",
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
    }


def build_generic_tool_resource(
    warehouse: str,
    identifier: str,
    query_timeout: int = 60,
) -> Dict[str, Any]:
    """Helper to build a generic tool resource (UDF/stored procedure)."""
    return {
        "type": "function",
        "execution_environment": {
            "type": "warehouse",
            "warehouse": warehouse,
            "query_timeout": query_timeout,
        },
        "identifier": identifier,
    }


def build_cortex_analyst_tool(
    name: str,
    description: str,
) -> Dict[str, Any]:
    """Helper to build a Cortex Analyst tool specification."""
    return {
        "tool_spec": {
            "type": "cortex_analyst_text_to_sql",
            "name": name,
            "description": description,
        }
    }


def build_cortex_analyst_resource(
    warehouse: str,
    semantic_model_file: Optional[str] = None,
    semantic_view: Optional[str] = None,
    query_timeout: int = 60,
) -> Dict[str, Any]:
    """Helper to build a Cortex Analyst tool resource."""
    resource = {
        "execution_environment": {
            "type": "warehouse",
            "warehouse": warehouse,
            "query_timeout": query_timeout,
        }
    }

    if semantic_model_file:
        resource["semantic_model_file"] = semantic_model_file
    elif semantic_view:
        resource["semantic_view"] = semantic_view
    else:
        raise ValueError(
            "Must provide either semantic_model_file or semantic_view"
        )

    return resource


def build_cortex_search_tool(
    name: str,
    description: str,
) -> Dict[str, Any]:
    """Helper to build a Cortex Search tool specification."""
    return {
        "tool_spec": {
            "type": "cortex_search",
            "name": name,
            "description": description,
        }
    }


def build_cortex_search_resource(
    search_service: str,
    title_column: str,
    id_column: str,
    filter_query: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper to build a Cortex Search tool resource."""
    resource = {
        "search_service": search_service,
        "title_column": title_column,
        "id_column": id_column,
    }

    if filter_query:
        resource["filter"] = filter_query

    return resource
