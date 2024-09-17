from hypster import HP

def my_config(hp: HP):
    data_path = hp.text_input("data")
    env = hp.select(["dev", "prod"], default="dev")

    llm_model = hp.select(
        {
            "haiku": "claude-3-haiku-20240307",
            "sonnet": "claude-3-5-sonnet-20240620",
        },
        default="haiku",
    )