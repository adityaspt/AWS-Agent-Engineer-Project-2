"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Fill these in with the values you collected during infrastructure setup
# (phase2_outputs.json / the AWS console), before your first deploy.
GATEWAY_URL = "https://customersupportgateway-yqfybu12lj.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "E0D7XTWQUH"          # 10-character Knowledge Base ID
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-ejQHij3w65"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    namespaces: Dict = {}
    strategies = mem_client.get_memory_strategies(memory_id)
    for strategy in strategies:
        strategy_type = strategy.get("type") or strategy.get("strategyType")
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]
    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    @staticmethod
    def _is_tool_result(content) -> bool:
        return any(isinstance(block, dict) and "toolResult" in block for block in content)

    @staticmethod
    def _extract_text(content) -> str:
        parts = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
        return "\n".join(parts)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        content = last_message.get("content", [])

        # Only run for plain-text user messages, never for tool results
        # (Strands delivers tool results with role "user" too).
        if last_message.get("role") != "user" or self._is_tool_result(content):
            return

        user_query = self._extract_text(content)
        if not user_query:
            return

        context_lines = []
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning("Memory retrieval failed for namespace %s: %s", namespace, e)
                continue

            for memory in memories or []:
                text = (memory.get("content") or {}).get("text") or memory.get("text")
                if text:
                    context_lines.append(f"[{strategy_type}] {text}")

        if context_lines:
            context_block = "\n".join(context_lines)
            new_text = f"Customer Context:\n{context_block}\n\n{user_query}"
            last_message["content"] = [{"text": new_text}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        customer_query = None
        agent_response = None

        for message in reversed(messages):
            role = message.get("role")
            content = message.get("content", [])
            if self._is_tool_result(content):
                continue
            text = self._extract_text(content)
            if not text:
                continue
            if role == "assistant" and agent_response is None:
                agent_response = text
            elif role == "user" and customer_query is None:
                customer_query = text
            if customer_query is not None and agent_response is not None:
                break

        if customer_query is None or agent_response is None:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning("Failed to save support interaction to memory: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.warning("Knowledge base retrieve failed: %s", e)
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

# 500 points = $5 of redemption value. Floor to the nearest 500 points,
# and cap the redeemed value at 50% of the order total.
max_redeemable_value = order_total * 0.5
raw_points_value = (loyalty_points // 500) * 5.0
points_redeemed_value = min(raw_points_value, max_redeemable_value)
points_redeemed = int((points_redeemed_value / 5.0) * 500) if points_redeemed_value > 0 else 0
points_redeemed_value = (points_redeemed / 500) * 5.0 if points_redeemed else 0.0

subtotal_after_points = order_total - points_redeemed_value

tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount_amount = subtotal_after_points * tier_discount_pct

final_total = round(subtotal_after_points - tier_discount_amount, 2)
total_savings = round(order_total - final_total, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_redeemed_value": round(points_redeemed_value, 2),
    "tier_discount_pct": tier_discount_pct,
    "tier_discount_amount": round(tier_discount_amount, 2),
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response["stream"]:
                result = event["result"]
                return json.dumps(result)
            return json.dumps({"error": "No result returned from code interpreter"})

    except Exception as e:
        logger.warning("Code interpreter unavailable, using fallback: %s", e)
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        final_total = round(order_total * (1 - tier_discount_pct), 2)
        return json.dumps(
            {
                "points_redeemed": 0,
                "tier_discount_pct": tier_discount_pct,
                "final_total": final_total,
                "remaining_points": loyalty_points,
                "note": "Code interpreter unavailable; tier-only discount applied as fallback.",
            }
        )


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id") or "default_customer"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

        with gateway_client:
            gateway_tools = gateway_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a helpful, concise customer support assistant for an "
                    "e-commerce platform. Use the order-tracking and refund tools "
                    "for order/refund questions, the knowledge base tool for product "
                    "and policy questions, the loyalty discount tool for any discount "
                    "or points calculation, and the browser tool when the customer "
                    "asks about a live web page. Always double-check IDs before "
                    "calling a tool, and never fabricate order, refund, or pricing "
                    "details — rely on the tools for factual answers."
                ),
            )

            response = agent(user_input)

        return response.message["content"][0]["text"]

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"An error occurred while processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
