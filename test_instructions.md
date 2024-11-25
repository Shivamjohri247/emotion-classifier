EMOTION CLASSIFICATION SYSTEM - TEST INSTRUCTIONS
==============================================

QUICK START
----------
1. Install dependencies:
   pip install -e ".[dev]"

2. Run tests:
   pytest

TEST STRUCTURE
-------------
```
tests/
├── config/              - Test configurations
├── data/               - Test data
├── unit/               - Unit tests
├── integration/        - Integration tests
└── conftest.py         - Shared fixtures
```
BASIC COMMANDS
-------------
1. All tests:
   pytest

2. Specific tests:
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/unit/test_data_ingestion.py

3. Coverage:
   pytest --cov=src --cov-report=html

TEST COMPONENTS
--------------
1. Data Ingestion Tests
   - Data loading
   - Preprocessing
   - Train-test split

2. Model Training Tests
   - Model initialization
   - Training process
   - Model saving

3. API Tests
   - Endpoints
   - Request handling
   - Response format

WRITING TESTS
------------
1. Create test file:
   tests/unit/test_new_feature.py

2. Basic test structure:
   def test_feature():
       # Arrange
       expected = "value"
       
       # Act
       result = function()
       
       # Assert
       assert result == expected

DEBUGGING
---------
1. Print output:
   pytest -v --capture=no

2. Stop on first fail:
   pytest -x

3. Debug on fail:
   pytest --pdb

COMMON ISSUES
------------
1. Missing Dependencies
   - Run: pip install -e ".[dev]"

2. Data Missing
   - Run: python tests/data/create_test_data.py

3. CUDA Errors
   - Reduce batch size in test_config.yaml

MAINTENANCE
----------
1. Keep test data updated
2. Run tests before commits
3. Maintain coverage levels
4. Update documentation

CONTACT
-------
Maintainer: ShivamJohri
