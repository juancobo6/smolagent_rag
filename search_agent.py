from smolagents import LiteLLMModel, CodeAgent, DuckDuckGoSearchTool


def main():
    model = LiteLLMModel(
        model_id = "ollama_chat/llama3.2:1b",
    )

    search_tool = DuckDuckGoSearchTool()

    agent = CodeAgent(
        tools=[search_tool], model=model, max_iterations=4, verbose=True
    )

    agent_output = agent.run("Use the search tool to retrieve information about Ancient Greece and make an essay explaining the origins of the city-state.")

    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    main()

    print(0)