# src/privacyraven/console.py
import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
def main():
    # TODO: Add ability to run attacks from checkpoint file
    click.echo("Hello, world!")
