# GET /trips returns the 10 evaluator trip IDs (no labels).
# POST /evaluate/{trip_id} runs the full prediction pipeline and returns the predicted route,
# then calls evaluation_service to score it against server-side labels.
# Owner: A.
