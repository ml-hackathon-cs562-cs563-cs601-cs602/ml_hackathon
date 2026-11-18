# Load all trips for a partition into memory with a single function call.
# Used by pipelines and the API. Only load .route for training and student trips here.
# Evaluator .route loading happens only in api/evaluation_service.py.
# Owner: A.
