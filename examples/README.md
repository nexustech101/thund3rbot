# Thund3rBot Examples

These examples show common ways to embed Thund3rBot in application code.

Most examples use small scripted models so they can run without provider API
keys. Examples that require extras call that out at the top of the file.

## Examples

- `local_cli_chat.py` - minimal interactive CLI using a local/provider model.
- `typed_sentiment_service.py` - structured sentiment output with a fake model.
- `approval_hooks_finance.py` - approval hooks for high-risk transactions.
- `webpage_extraction_tool.py` - webpage-to-artifact extraction with a tool.
- `fastapi_agent_route.py` - FastAPI router integration.
- `registers_agent_cli.py` - minimal URL-to-agent-to-database ecommerce extractor.

Run from the repository root:

```bash
python examples/typed_sentiment_service.py
python examples/approval_hooks_finance.py
python examples/webpage_extraction_tool.py
```

The registers example requires the `registers` package:

```bash
pip install "thund3rbot[registers,cli]"
python examples/registers_agent_cli.py extract "https://shop.example/product"
```

Provider-backed examples require the matching extra:

```bash
pip install "thund3rbot[providers]"
pip install "thund3rbot[fastapi]"
```
