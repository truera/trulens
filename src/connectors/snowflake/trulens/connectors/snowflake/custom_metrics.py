"""Custom metrics management for Snowflake connector."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from trulens.connectors.snowflake.dao.sql_utils import execute_query

logger = logging.getLogger(__name__)


class CustomMetricsManager:
    """Manages custom metrics registration and metadata in Snowflake."""

    def __init__(self, snowpark_session):
        """Initialize the custom metrics manager.

        Args:
            snowpark_session: Active Snowpark session
        """
        self.session = snowpark_session

    def register_custom_metric(
        self,
        metric_name: str,
        metric_type: str,
        computation_type: str,
        object_name: str,
        object_type: str,
        object_version: str,
        selectors: Dict[str, Any],
        description: Optional[str] = None,
    ) -> None:
        """
        Register a custom metric definition with Snowflake.

        Args:
            metric_name: Name of the metric
            metric_type: Type identifier for the metric
            computation_type: "client" or "server"
            object_name: Name of the managing object
            object_type: Type of the managing object
            object_version: Version of the managing object
            selectors: Dictionary of selectors for the metric
            description: Optional description of the metric
        """
        metric_definition = {
            "name": metric_name,
            "type": metric_type,
            "computation_type": computation_type,
            "selectors": self._serialize_selectors(selectors),
            "description": description,
        }

        try:
            # Check if custom metrics table exists, create if not
            self._ensure_custom_metrics_table()

            # Insert or update the metric definition
            query = """
            MERGE INTO CUSTOM_METRICS_REGISTRY AS target
            USING (
                SELECT ? AS object_name, ? AS object_type, ? AS object_version,
                       ? AS metric_name, ? AS metric_definition
            ) AS source
            ON target.OBJECT_NAME = source.object_name
               AND target.OBJECT_TYPE = source.object_type
               AND target.OBJECT_VERSION = source.object_version
               AND target.METRIC_NAME = source.metric_name
            WHEN MATCHED THEN
                UPDATE SET METRIC_DEFINITION = source.metric_definition,
                          UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (OBJECT_NAME, OBJECT_TYPE, OBJECT_VERSION, METRIC_NAME, METRIC_DEFINITION, CREATED_AT, UPDATED_AT)
                VALUES (source.object_name, source.object_type, source.object_version,
                       source.metric_name, source.metric_definition, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
            """

            execute_query(
                self.session,
                query,
                parameters=(
                    object_name,
                    object_type,
                    object_version,
                    metric_name,
                    json.dumps(metric_definition),
                ),
            )

            logger.info(f"Registered custom metric: {metric_name}")

        except Exception as e:
            logger.error(f"Failed to register custom metric {metric_name}: {e}")
            raise

    def get_available_metrics(
        self,
        object_name: str,
        object_type: str,
        object_version: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all available metrics for an object.

        Args:
            object_name: Name of the managing object
            object_type: Type of the managing object
            object_version: Version of the managing object

        Returns:
            List of metric definitions
        """
        try:
            query = """
            SELECT METRIC_NAME, METRIC_DEFINITION
            FROM CUSTOM_METRICS_REGISTRY
            WHERE OBJECT_NAME = ? AND OBJECT_TYPE = ? AND OBJECT_VERSION = ?
            """

            rows = execute_query(
                self.session,
                query,
                parameters=(object_name, object_type, object_version),
            )

            metrics = []
            for row in rows:
                try:
                    definition = json.loads(row["METRIC_DEFINITION"])
                    metrics.append({
                        "name": row["METRIC_NAME"],
                        "definition": definition,
                    })
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse metric definition for {row['METRIC_NAME']}: {e}"
                    )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get available metrics: {e}")
            return []

    def delete_custom_metric(
        self,
        metric_name: str,
        object_name: str,
        object_type: str,
        object_version: str,
    ) -> bool:
        """
        Delete a custom metric definition.

        Args:
            metric_name: Name of the metric to delete
            object_name: Name of the managing object
            object_type: Type of the managing object
            object_version: Version of the managing object

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            query = """
            DELETE FROM CUSTOM_METRICS_REGISTRY
            WHERE OBJECT_NAME = ? AND OBJECT_TYPE = ? AND OBJECT_VERSION = ? AND METRIC_NAME = ?
            """

            execute_query(
                self.session,
                query,
                parameters=(
                    object_name,
                    object_type,
                    object_version,
                    metric_name,
                ),
            )

            logger.info(f"Deleted custom metric: {metric_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete custom metric {metric_name}: {e}")
            return False

    def _ensure_custom_metrics_table(self) -> None:
        """Ensure the custom metrics registry table exists."""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS CUSTOM_METRICS_REGISTRY (
                OBJECT_NAME VARCHAR(255) NOT NULL,
                OBJECT_TYPE VARCHAR(255) NOT NULL,
                OBJECT_VERSION VARCHAR(255) NOT NULL,
                METRIC_NAME VARCHAR(255) NOT NULL,
                METRIC_DEFINITION VARIANT NOT NULL,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (OBJECT_NAME, OBJECT_TYPE, OBJECT_VERSION, METRIC_NAME)
            )
            """

            execute_query(self.session, create_table_query)
            logger.debug("Custom metrics registry table ensured")

        except Exception as e:
            logger.error(f"Failed to create custom metrics registry table: {e}")
            raise

    def _serialize_selectors(self, selectors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize selectors for storage in Snowflake.

        Args:
            selectors: Dictionary of selectors

        Returns:
            Serialized selectors dictionary
        """
        serialized = {}

        for key, selector in selectors.items():
            if hasattr(selector, "__dict__"):
                # Convert selector object to dictionary
                serialized[key] = {
                    "function_name": getattr(selector, "function_name", None),
                    "span_name": getattr(selector, "span_name", None),
                    "span_type": getattr(selector, "span_type", None),
                    "span_attribute": getattr(selector, "span_attribute", None),
                    "function_attribute": getattr(
                        selector, "function_attribute", None
                    ),
                    "trace_level": getattr(selector, "trace_level", False),
                    "collect_list": getattr(selector, "collect_list", True),
                    "ignore_none_values": getattr(
                        selector, "ignore_none_values", False
                    ),
                    "match_only_if_no_ancestor_matched": getattr(
                        selector, "match_only_if_no_ancestor_matched", False
                    ),
                }
            else:
                # Assume it's already serializable
                serialized[key] = selector

        return serialized
