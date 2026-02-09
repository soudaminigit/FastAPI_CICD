.github/workflows contain CI/CD workflows.
ci-cd.yml -> Loads the code, creates docker and pushes to docker hub
cd-drift.yml -> Check out code, run data drift with real data and trigger retraining of the model, if required.
It compares the differemce between current_data.tsv and reference_data.tsv
model_mlflow -> Runs the model and saves the model artifacts
