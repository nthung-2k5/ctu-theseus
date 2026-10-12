"""A remote training worker: trains runs on this machine for an OmniTrain server.

    OMNITRAIN_API_URL=https://omnitrain.example OMNITRAIN_WORKER_TOKEN=thwk_... python -m omnitrain.worker

It pulls. Every few seconds it asks the API (`/api/worker/claim`) for a queued run, trains it with exactly the code the
server uses for local runs (`omnitrain.jobs.run_core`), and reports progress and the outcome back over HTTP. It never
opens a database connection: the run's data and results go through S3 directly (S3_ENDPOINT, S3_ACCESS_KEY and
S3_SECRET_KEY must reach the same buckets the server uses), and everything else goes through the API, which stays the
only writer of run state.
"""
