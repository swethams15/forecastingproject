# forecastingproject
# Setting up Dev env for Python.

## Tools required.
- Python 3.9
- venv or conda

# Setup
- Create a venv or conda env and activate it.
- run the command pip install `pip install -r prediction-gateway/requirements-dev.txt`

# Starting prediction-gateway server
- `cd prediction-gateway`
- `python3 app.py` This should start the flask server in the development mode.

# Starting the prediction-gateway in Production mode.
- `cd prediction-gateway`
- `gunicorn --bind 0.0.0.0:5000 wsgi:app` This should start the gunicorn server in production mode at port 5000

# Starting the prediction-gateway on Production server
- On the root folder run the command `docker compose up -d`

## Committing the code.

- Ensure there are no lint errors by running `yarn lint`.
- Ensure there are no build errors by running `yarn build`.
- The commit message should have the format "type: message"
- Must be one of the following:
  - test — Adding missing tests
  - feat — A new feature
  - fix — A bug fix
  - chore — Build process or auxiliary tool changes
  - docs — Documentation only changes
  - refactor — A code change that neither fixes a bug or adds a feature
  - style — Markup, white-space, formatting, missing semi-colons...
  - ci — CI related changes
  - perf — A code change that improves performance
- Push the code to the remote repository and raise a PR.
- Ensure that the branch is deleted after the PR is merged.

# Codegen and Stylegen

- Run the command `yarn codegen` when there is a change in the graphql.
- Run the command `yarn gen-css-types` when there is change in stylesheets.
