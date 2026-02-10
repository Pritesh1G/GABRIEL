import asyncio
import pandas as pd
from gabriel.utils import openai_utils


def test_get_all_responses_reset_files(tmp_path):
    save_path = tmp_path / "out.csv"
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b"],
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
        )
    )
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["b"],
            identifiers=["2"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=True,
        )
    )
    assert set(df["Identifier"]) == {"2"}


def test_resume_treats_string_success_values_as_completed(tmp_path):
    save_path = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Identifier": ["1", "2", "3"],
            "Response": ["[]", "[]", "[]"],
            "Web Search Sources": ["[]", "[]", "[]"],
            "Time Taken": [0.1, 0.1, 0.1],
            "Input Tokens": [1, 1, 1],
            "Reasoning Tokens": [0, 0, 0],
            "Output Tokens": [1, 1, 1],
            "Reasoning Effort": ["default", "default", "default"],
            "Successful": ["True", "true", "1"],
            "Error Log": ["[]", "[]", "[]"],
            "Response IDs": ["[]", "[]", "[]"],
            "Reasoning Summary": ["", "", ""],
        }
    ).to_csv(save_path, index=False)

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b", "c"],
            identifiers=["1", "2", "3"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=False,
        )
    )

    assert len(df) == 3
