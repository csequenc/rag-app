def build_tool_descriptions(tools) -> str:
    return "\n\n".join(
        f"{tool.name}\n{tool.description}"
        for tool in tools
    )