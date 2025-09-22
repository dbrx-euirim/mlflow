"""
Search traces tool for MLflow GenAI judges.

This module provides a tool for retrieving the ids of traces that share the same session id.
"""

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class SearchTracesTool(JudgeTool):
    """
    Tool for retrieving the ids of traces that share the same session id.
    """

    @property
    def name(self) -> str:
        return ToolNames.SEARCH_TRACES

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.SEARCH_TRACES,
                description=(
                    "Retrieve the ids of traces that share the same session id. This is useful for "
                    "finding all traces that were part of the same session."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace) -> TraceInfo:
        """
        Get metadata about the trace.

        Args:
            trace: The MLflow trace object to analyze

        Returns:
            TraceInfo object
        """
        return trace.info
