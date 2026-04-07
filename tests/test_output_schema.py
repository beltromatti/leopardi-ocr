from leopardi.schemas.output import ParsedPage


def test_example_schema_contains_markdown_and_blocks() -> None:
    page = ParsedPage.example()
    assert page.markdown.startswith("# Leopardi")
    assert len(page.blocks) == 2
    assert page.blocks[0].markdown == "# Leopardi"
