# TruLens Dashboard Streamlit Tests

> Unit tests for TruLens Dashboard Streamlit components using the AppTest framework.

## Quick Start

```bash
# Run all streamlit tests
pytest tests/unit/streamlit/

# Run with coverage report
pytest tests/unit/streamlit/ --cov=trulens.dashboard.streamlit --cov-report=html
```

## 📋 Table of Contents

- [Test Files Overview](#test-files-overview)
- [Current Coverage](#current-coverage)
- [Testing Gaps & Priorities](#testing-gaps--priorities)
- [Contributing](#contributing)
- [Test Utilities Reference](#test-utilities-reference)

## Test Files Overview

| File | Purpose | Key Features |
|------|---------|--------------|
| `test_streamlit_components.py` | Core component functionality | Component rendering, data handling, error states |
| `test_streamlit_interactions.py` | UI interactions & forms | Button clicks, form submission, state management |
| `test_streamlit_leaderboard.py` | Leaderboard-specific features | Page rendering, data display, navigation |
| `test_streamlit_utils.py` | Shared test utilities | Mocks, factories, helpers, scenarios |

## Current Coverage

### ✅ Well Tested
- **Component Rendering**: All major UI components render correctly
- **Data Scenarios**: Empty states, populated data, large datasets
- **Basic Interactions**: Buttons, forms, toggles, selections
- **Error Handling**: Basic error states and user feedback
- **State Management**: Session state and component initialization

### ⚠️ Partially Tested
- **Feedback Systems**: Basic display but not complex interactions
- **Data Processing**: Mock data handling but not real performance
- **UI Controls**: Presence testing but limited interaction testing

## Testing Gaps & Priorities

### 🔴 Critical (High Impact, Missing)

| Gap | Impact | Effort |
|-----|---------|---------|
| **Page Navigation** | High | Medium |
| **Authentication Flow** | High | High |
| **Real-time Updates** | High | Medium |
| **Performance with Large Data** | High | Medium |

### 🟡 Important (Medium Priority)

| Gap | Impact | Effort |
|-----|---------|---------|
| **Advanced UI Interactions** | Medium | Low |
| **End-to-end Workflows** | Medium | High |
| **Cross-browser Testing** | Medium | High |
| **Error Edge Cases** | Medium | Medium |

### 🟢 Nice to Have (Lower Priority)

- Accessibility compliance
- Internationalization 
- Visual regression testing
- Mobile responsiveness

## Next Steps Roadmap

### Phase 1: Foundation (Next Sprint)
1. **Add navigation tests** - Test `st.switch_page()` functionality
2. **Mock real-time features** - Test auto-refresh components
3. **Performance benchmarks** - Baseline metrics for large datasets

### Phase 2: Integration (Next Month)
1. **End-to-end scenarios** - Complete user workflows
2. **Authentication testing** - Database connection scenarios
3. **Cross-page state** - URL parameters and deep linking

### Phase 3: Polish (Future)
1. **Browser compatibility** - Selenium/Playwright integration
2. **Accessibility** - Screen reader and keyboard navigation
3. **Load testing** - Concurrent user scenarios

## Contributing

### Adding New Tests

1. **Use existing utilities** from `test_streamlit_utils.py`
2. **Follow naming conventions**: `test_[component]_[scenario]`
3. **Include positive and negative cases**
4. **Mock external dependencies**
5. **Update this README** when filling gaps

### Example Test Structure

```python
def test_component_with_scenario(self, mock_data):
    """Test description explaining what this verifies."""
    with MockManager.mock_dashboard_utils(mock_data):
        def test_app():
            # Your component test code
            pass
        
        app = AppTestHelper.create_and_run_app(test_app)
        AppTestHelper.assert_no_errors(app)
        # Add specific assertions
```

## Test Utilities Reference

### Core Classes

- **`TestDataFactory`** - Creates realistic mock data
  - `create_records_df()` - Mock database records
  - `create_feedback_defs()` - Mock feedback definitions
  - `create_large_dataset()` - Performance testing data

- **`AppTestHelper`** - Test execution and assertions
  - `create_and_run_app()` - Execute Streamlit app tests
  - `assert_has_*()` - UI element presence checks
  - `assert_no_errors()` - Error-free execution verification

- **`MockManager`** - Centralized mocking
  - `mock_tru_session()` - Mock TruLens session
  - `mock_dashboard_utils()` - Mock dashboard utilities
  - `mock_all_common_dependencies()` - Full stack mocking

### Quick Reference

```python
# Basic test pattern
with MockManager.mock_tru_session(mock_data=data):
    app = AppTestHelper.create_and_run_app(test_function)
    AppTestHelper.assert_no_errors(app)

# Create test data
data = create_mock_data_dict(app_ids=["test_app"], size=10)

# Test error scenarios
app = AppTestHelper.create_and_run_app(test_function, should_raise=True)
```

## Technical Notes

- **Framework**: `streamlit.testing.v1.AppTest`
- **Mocking Strategy**: External dependencies mocked for isolation
- **Focus**: Component behavior over visual appearance
- **Integration**: Separate from unit tests, requires different setup

---

💡 **Tip**: Start with existing test patterns in `test_streamlit_utils.py` before creating new testing approaches. 