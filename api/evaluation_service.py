# The ONLY module allowed to read evaluator ground truth (api/server_data/).
# Computes accuracy and route metrics after prediction. Must never be imported by
# src/pathsnap/ or pipelines/. Owner: A.
