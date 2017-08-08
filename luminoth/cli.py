import click

from .tools import cloud, dataset
from .train import train
from .eval import evaluate


@click.group()
def cli():
    pass


cli.add_command(cloud)
cli.add_command(dataset)
cli.add_command(train)
cli.add_command(evaluate)
