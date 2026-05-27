"""
This script defines a simple CLI tool that fetches raw HTML from an
ecommerce URL, sends that HTML to a Thund3rBot agent for structured
product extraction, and bulk-saves the extracted JSON data to a local
SQLite database using registers.db.

Run:
    python examples/registers_agent_cli.py extract "https://example.com/product"

Requirements:
    pip install "thund3rbot[registers,providers,cli]"
"""
from __future__ import annotations

# 1. Import necessary libraries and modules
import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field
import registers.cli as cli
from registers import database_registry, db_field, dispose_all
from rich.console import Console
from rich.table import Table

from thund3rbot import AgentFactory, AgentScope, AgentSpec, FactoryConfig, ModelConfig


# 2. Configure local database and extraction schemas
DB_PATH = Path(__file__).with_suffix(".db")
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"
console = Console()


class ExtractedProduct(BaseModel):
    """Structured product data returned by the agent."""

    title: str = ""
    price: str = ""
    currency: str = ""
    availability: str = ""
    sku: str = ""
    brand: str = ""
    description: str = ""
    image_urls: list[str] = Field(default_factory=list)


class ProductExtraction(BaseModel):
    """Wrapper schema so the agent can return one or many products."""

    products: list[ExtractedProduct] = Field(default_factory=list)


@database_registry(DB_URL, table_name="products", key_field="id")
class ProductRecord(BaseModel):
    """Database model used to persist extracted product data."""
    id: int | None = db_field(default=None, id_strategy="autoincrement")
    created_at: str = db_field(index=True)
    source_url: str = db_field(index=True)
    title: str = db_field(index=True)
    price: str = ""
    currency: str = ""
    availability: str = ""
    sku: str = ""
    brand: str = ""
    description: str = ""
    image_urls_json: str = "[]"


# Helper function to build data table object
def build_products_table(products: list[ExtractedProduct]) -> Table:
    """Print a rich table of extracted products."""
    table = Table(title="Extracted Products")
    table.add_column("#", justify="right")
    table.add_column("Title")
    table.add_column("Price")
    table.add_column("Currency")
    table.add_column("Brand")
    table.add_column("Created At")
    for product in products:
        table.add_row(
            str(product.id),
            product.title,
            product.price,
            product.currency,
            product.brand or "NULL",
            product.created_at,
        )
    return table


# Helper function to print data table 
def print_products_table(products: list[ExtractedProduct]) -> None:
    """Print a rich table of extracted products."""
    table = build_products_table(products)
    console.print(table)


# Helper function to print data table as json
def print_products_json(products: list[ExtractedProduct]) -> None:
    """Print extracted products as JSON."""
    console.print_json(json.dumps([product.model_dump(mode="json") for product in products], indent=2))


# 3. Define a function to fetch raw HTML from any URL
def fetch_html(url: str) -> str:
    """
    Fetch and return raw HTML from a URL.

    Args:
        url (str): The page URL to fetch.

    Returns:
        str: Raw HTML from the response body.
    """
    with console.status(f"[bold blue]Fetching webpage[/] {url}", spinner="dots"):
        request = Request(url, headers={"User-Agent": "thund3rbot-example/1.0"})
        with urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")

    console.print(f"[green]Fetched[/] {len(html):,} characters of HTML")
    return html


# 4. Define a function to create the extraction agent
@lru_cache(maxsize=1)
def create_extraction_agent(
    *,
    provider: str = "ollama",
    model: str = "qwen3.5:9b",
):
    """
    Create and return an agent configured for ecommerce extraction.

    Args:
        provider (str): The language model provider to use.
        model (str): The specific model to use from the provider.

    Returns:
        An AgentFactory agent configured to return ProductExtraction.
    """
    with console.status(
        f"[bold blue]Configuring agent[/] provider={provider} model={model}",
        spinner="dots",
    ):
        framework = AgentFactory(
            FactoryConfig(
                default_model=ModelConfig(
                    provider=provider,
                    model=model,
                    extra_kwargs={
                        "keep_alive": "1h",  # or "-1" to keep loaded indefinitely
                    },
                )
            )
        )

        agent = framework.agent(
            AgentSpec(
                name="ecommerce_extractor",
                scope=AgentScope.TASK,
                instructions=(
                    "You extract product data from raw ecommerce HTML. "
                    "Return only JSON matching this schema: "
                    '{"products": [{"title": "", "price": "", "currency": "", '
                    '"availability": "", "sku": "", "brand": "", '
                    '"description": "", "image_urls": []}]}. '
                    "If the page contains multiple products, return all of them. "
                    "If a field is missing, use an empty string or empty list."
                ),
                output_schema=ProductExtraction,
            )
        )

    console.print("[green]Agent ready[/]")
    return agent


# 5. Define a function to bulk-save extracted products
def save_products(url: str, products: list[ExtractedProduct]) -> list[ProductRecord]:
    """
    Save extracted products to the database in one bulk operation.

    Args:
        url (str): Source URL used for extraction.
        products (list[ExtractedProduct]): Products returned by the agent.

    Returns:
        list[ProductRecord]: Persisted database records.
    """
    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "created_at": now,
            "source_url": url,
            "title": product.title,
            "price": product.price,
            "currency": product.currency,
            "availability": product.availability,
            "sku": product.sku,
            "brand": product.brand,
            "description": product.description,
            "image_urls_json": json.dumps(product.image_urls),
        }
        for product in products
    ]
    with console.status(f"[bold blue]Saving[/] {len(rows)} product record(s)", spinner="dots"):
        saved = ProductRecord.objects.bulk_create(rows)

    console.print(f"[green]Saved[/] {len(saved)} product record(s) to {DB_PATH}")
    return saved


# 6. Define the registers.cli command
@cli.register(
    name="extract",
    description="Fetch a product page, extract product JSON with an agent, and save it",
    examples=[
        'python examples/registers_agent_cli.py extract "https://example.com/product"',
        (
            'python examples/registers_agent_cli.py extract "https://example.com/product" '
            "openai gpt-4o-mini"
        ),
    ],
)
@cli.argument("url", type=str, help="Ecommerce product page URL")
@cli.argument("provider", type=str, default="ollama", help="Model provider")
@cli.argument("model", type=str, default="qwen3.5:9b", help="Model name")
@cli.alias("--extract")
@cli.alias("-e")
async def extract(url: str, provider: str = "ollama", model: str = "qwen3.5:9b") -> None:
    """
    Fetch HTML, run the extraction agent, and save product records.

    Args:
        url (str): Ecommerce product page URL.
        provider (str): Language model provider.
        model (str): Language model name.
    """
    console.rule("[bold]Thund3rBot Ecommerce Extractor")
    console.print(f"[cyan]URL[/] {url}")
    console.print(f"[cyan]Model[/] {provider}/{model}")

    html = fetch_html(url)
    agent = create_extraction_agent(provider=provider, model=model)

    with console.status("[bold blue]Extracting product data with agent[/]", spinner="dots"):
        result = await agent.run(
            f"Source URL: {url}\n\nRaw HTML:\n{html}",
            context={"source_url": url, "html_length": len(html)},
        )
    if result.error:
        console.print(f"[red]Agent error:[/] {result.error}")
        raise SystemExit(1)

    extraction: ProductExtraction = result.output
    console.print(f"[green]Extracted[/] {len(extraction.products)} product(s)")
    saved = save_products(url, extraction.products)

    print_products_table(saved)
    print_products_json(saved)


@cli.register("list", description="List all extracted products in the database")
@cli.alias("--list")
@cli.alias("-l")
def list_products() -> None:
    """Helper function to list all saved products in the database as json."""
    # console.rule("[bold]All Extracted Products")
    records = ProductRecord.objects.all()
    # print_products_table(records)
    print_products_json(records)


# 7. Run the CLI when the script is executed
if __name__ == "__main__":
    try:
        cli.run(
            shell_title="Ecommerce Extractor",
            shell_description="Fetch product HTML, extract JSON with an agent, and save to SQLite.",
            shell_usage=True,
        )
    finally:
        dispose_all()
