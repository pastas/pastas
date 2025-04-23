Using AI to Generate Unit Tests for Pastas
=========================================

Writing unit tests for Pastas can be time consuming and, let's be honest, a bit boring. Tests are critical, however, to ensure that the code works correctly when making further changes in the future. Artificial intelligence (AI) tools can help automate the generation of unit tests, making it easier to maintain high test coverage and ensure code quality. This guide provides recommendations for using AI tools (like GitHub Copilot or Cursor) to assist with creating and improving unit tests for the Pastas library, particularly when adding new functions and modules.

Getting Started with AI-Assisted Testing
---------------------------------------

Prerequisites
~~~~~~~~~~~~

- Understanding of Pastas core functionality
- Access to an AI coding assistant (GitHub Copilot, ChatGPT, etc.)
- Familiarity with the Python testing framework Pytest helps

Introduction
~~~~~~~~~~~~

The test suite for Pastas is located in the ``tests`` directory. It contains a variety of tests, including unit tests, integration tests, and regression tests. The goal is to ensure that all components of Pastas work as expected and that any changes made to the code do not introduce new bugs. Pastas uses the Pytest framework for testing, which is a powerful and flexible testing tool for Python. The tests are automatically run using GitHub Actions, ensuring that the code is continuously tested and validated.

Before using AI to generate tests, it's important to write code that is easy to test. This means:

- Writing modular code with clear interfaces
- Type hinting for function parameters and return values
- Using docstrings to explain the purpose and behavior of functions
- Following the Pastas code style guide

Following the above practices will make it easier for AI to understand the code and generate meaningful tests. After writing the code, you can use AI to help generate tests that cover various scenarios, including edge cases and error handling.

Setup
~~~~~

After writing your code, we can start adding unit tests. There are many ways to ask AI to generate tests. The most common way is to use the AI's code generation capabilities to create a test function based on the code you have written. This can be done by providing the AI with a prompt that describes the function you want to test and the expected behavior.

Many modern IDEs (like PyCharm, VSCode, etc.) have built-in support for AI code generation. You can also use standalone AI tools like ChatGPT or GitHub Copilot. The key is to provide the AI with enough context about the function you want to test and the expected behavior, which is where the IDEs shine.

In VSCode, you can open the file you want to test and use the AI code generation feature to create a test function. For example, if you have a function called ``calculate_sum``, you can ask the AI to generate a test function for it by typing:

.. code-block:: none

   "Generate a unit tests for the calculate_sum function using pytest"

This will prompt the AI to generate a test function that tests the ``calculate_sum`` function. You can then review the generated test function and make any necessary adjustments.

Often, the generated unit tests will actually not run immediately, so it is important to check the generated code, and run the test on your local machine. Sometimes the generated unit tests will cause errors than need to be fixed in your new code, but often the test itself needs to be fixed.

The file `conftest.py` contains the fixtures that are used in the tests. Fixtures are functions that provide a fixed baseline for tests to reliably and repeatedly execute. They are used to set up the environment for the tests and can be used to create mock objects, set up databases, or perform any other setup that is needed for the tests to run.
The fixtures are defined using the ``@pytest.fixture`` decorator, and can be used in the tests by passing them as arguments to the test functions. It is recommended to check the generated code and see if the AI has used the fixtures correctly. If not, you can adjust the generated code to use the fixtures.

The AI may also generate tests that are not relevant to the code you have written. In this case, you can simply delete the irrelevant tests and keep the ones that are relevant. It is important to review the generated tests and make sure they are relevant to the code you have written.

When using AI to generate tests, it is important to remember that the AI is not perfect and may not always generate the correct tests. It is important to review the generated tests and make sure they are correct before committing them to the codebase.

Common AI Testing Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~

While AI tools can significantly accelerate test creation, be aware of these common issues:

1. **Missing Imports**: AI often fails to include all necessary imports, especially for Pastas-specific modules.
2. **Incorrect Fixture Usage**: Generated tests may not correctly use available fixtures in conftest.py.
3. **Overly Complex Tests**: AI sometimes generates overly complex tests that test too much at once.
4. **Incomplete Edge Cases**: AI may miss important edge cases specific to time series data processing.
5. **Hardcoded Values**: Tests may contain hardcoded values instead of parametrized inputs.

To address these issues, always run the tests locally before committing and be prepared to make adjustments.

Effective Prompting Strategies
-----------------------------

1. **Provide Context**: Share the relevant module code or function signature you want to test

   .. code-block:: none

      "I need to write tests for this Pastas function: [paste function code]"

2. **Specify Test Requirements**: Clearly state what aspects should be tested

   .. code-block:: none

      "Generate pytest tests that verify the response function correctly handles different parameter inputs"

3. **Include Edge Cases**: Ask for tests that cover edge cases specific to time series analysis

   .. code-block:: none

      "Include tests for handling missing values, irregular time steps, and boundary conditions"

4. **Reference Existing Tests**: Point AI to existing test patterns in the codebase

   .. code-block:: none

      "Follow the testing pattern in test_rfuncs.py where we parametrize tests across all available response functions"

Best Practices
------------

DO:
~~~

- Review and understand all AI-generated tests before committing
- Ensure tests follow Pastas' existing conventions (naming, structure, etc.)
- Add meaningful assertions that validate behavior, not just execution
- Refactor generated tests to improve readability and maintainability
- Add comments explaining test logic that might not be immediately obvious

DON'T:
~~~~~~

- Accept tests that only exercise code without meaningful assertions
- Commit tests that depend on external resources without proper mocking
- Rely solely on AI without understanding the test's purpose
- Skip test validation because "the AI wrote it"

Example Workflow
--------------

1. **Identify Untested Functionality**:
   Locate a Pastas module or function lacking test coverage

2. **Request Initial Test Structure**:

   .. code-block:: none

      "Generate a pytest test structure for testing the Exponential response function in Pastas"

3. **Refine the Tests**:

   .. code-block:: none

      "Add parameterized tests to check edge cases where alpha is very small or large"

4. **Review and Integrate**:

   - Manually review generated tests
   - Run tests and fix any issues
   - Refactor as needed
   - Add to the test suite

Example: Using AI to Create Parametrized Tests
--------------------------------------------

Here's an example of how you might prompt AI to create a parametrized test:

.. code-block:: none

   "Create a pytest.mark.parametrize test that verifies the Gamma response function in Pastas
   produces expected outputs for various combinations of parameters (n, A, a).
   Include edge cases and expected failure conditions."

Integration with Test Coverage Analysis
-------------------------------------

Use AI to help identify and address coverage gaps:

1. Run coverage analysis: ``pytest --cov=pastas``
2. Identify modules with low coverage
3. Ask AI to generate tests specifically targeting uncovered code paths

Maintenance and Updates
---------------------

As the Pastas codebase evolves:

1. Ask AI to review existing tests and suggest improvements
2. Generate additional tests for new features
3. Update outdated tests to match API changes

By following these guidelines, you can effectively leverage AI tools to improve Pastas' test coverage while maintaining code quality standards.
