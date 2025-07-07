import click
import json
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
@click.option('--config', prompt='Enter path to config JSON', required=True)
def main(config):

    with open(config, 'r') as f:
        data = json.load(f)

    run = ml_pipeline(data)
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == '__main__':
    main()
