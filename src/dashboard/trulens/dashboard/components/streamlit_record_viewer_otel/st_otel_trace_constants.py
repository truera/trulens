# Constants
STATUS_CODES = {"UNSET": "UNSET", "OK": "OK", "ERROR": "ERROR"}

STATUS_COLORS = {
    "UNSET": "#808080",  # Gray
    "OK": "#4CAF50",  # Green
    "ERROR": "#F44336",  # Red
}

# Duration Buckets Configuration
DURATION_BINS = [
    0,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1,
    5,
    10,
    float("inf"),
]  # in seconds
DURATION_LABELS = [
    "<100μs",
    "100-500μs",
    "500μs-1ms",
    "1-5ms",
    "5-10ms",
    "10-50ms",
    "50-100ms",
    "100-500ms",
    "500ms-1s",
    "1-5s",
    "5-10s",
    ">10s",
]

# Random Trace Generation Configuration
OPERATIONS = [
    "api_request",  # Root level operation
    "auth_check",  # Child of api_request
    "db_query",  # Child of api_request
    "cache_lookup",  # Child of db_query
    "payment_processing",  # Child of api_request
    "external_call",  # Child of api_request or db_query
    "file_read",  # Child of db_query
    "file_write",  # Child of db_query
    "message_queue",  # Child of api_request
    "notification_send",  # Child of payment_processing or api_request
]

OPERATION_HIERARCHY = {
    "api_request": [],
    "auth_check": ["api_request"],
    "db_query": ["api_request"],
    "cache_lookup": ["db_query"],
    "payment_processing": ["api_request"],
    "external_call": ["api_request", "db_query"],
    "file_read": ["db_query"],
    "file_write": ["db_query"],
    "message_queue": ["api_request"],
    "notification_send": ["payment_processing", "api_request"],
}

PARENT_SPAN_PROBABILITY = 0.7
MIN_DURATION = 0.01
MAX_DURATION = 3.0

# Helper to generate a random status code with more variety
STATUS_WEIGHTS = {
    "api_request": [0.7, 0.2, 0.1],
    "db_query": [0.6, 0.3, 0.1],
    "cache_lookup": [0.9, 0.09, 0.01],
    "auth_check": [0.8, 0.15, 0.05],
    "payment_processing": [0.7, 0.15, 0.15],
    "external_call": [0.7, 0.15, 0.15],
    "file_read": [0.85, 0.1, 0.05],
    "file_write": [0.85, 0.1, 0.05],
    "message_queue": [0.8, 0.15, 0.05],
    "notification_send": [0.8, 0.1, 0.1],
}

# Helper to generate a random duration based on operation
DURATION_RANGES = {
    "api_request": (0.5, 3.0),
    "db_query": (0.1, 1.5),
    "cache_lookup": (0.01, 0.2),
    "auth_check": (0.05, 0.5),
    "payment_processing": (0.2, 2.0),
    "external_call": (0.1, 1.0),
    "file_read": (0.01, 0.3),
    "file_write": (0.01, 0.3),
    "message_queue": (0.01, 0.5),
    "notification_send": (0.01, 0.5),
}

# --- Style Config ---
STYLE_BLOCK_BG = {
    "root": "#fff",
    "child": "#f3f4f8",  # Slightly higher contrast
}
STYLE_BLOCK_BORDER_RADIUS = "8px"  # More visible
STYLE_BLOCK_PADDING = "12px"  # More space
STYLE_BLOCK_MARGIN = "6px"
STYLE_STATUS_FONT_SIZE = "100%"
STYLE_ATTR_FONT_SIZE = "105%"
STYLE_EVENT_FONT_SIZE = "105%"
STYLE_TREE_MARKER_COLOR = "#444"  # Higher contrast
STYLE_DURATION_COLOR = "#444"
STYLE_NAME_FONT_WEIGHT = "bold"
STYLE_NAME_FONT_SIZE = "110%"

# --- Accessibility tweaks: higher contrast and larger font size ---
STYLE_BLOCK_BG = {
    "root": "#fff",
    "child": "#f3f4f8",  # Slightly higher contrast
}
STYLE_BLOCK_BORDER_RADIUS = "8px"  # More visible
STYLE_BLOCK_PADDING = "12px"  # More space
STYLE_BLOCK_MARGIN = "6px"
STYLE_STATUS_FONT_SIZE = "100%"
STYLE_ATTR_FONT_SIZE = "105%"
STYLE_EVENT_FONT_SIZE = "105%"
STYLE_TREE_MARKER_COLOR = "#444"  # Higher contrast
STYLE_DURATION_COLOR = "#444"
STYLE_NAME_FONT_WEIGHT = "bold"
STYLE_NAME_FONT_SIZE = "110%"
