import json
import pdb
import random

from pathlib import Path

import streamlit as st
import pandas as pd

from aletheia.utils import read_file


st.set_page_config(layout="wide")


HELP = """
Some notes:

- `is-valid` for real sentences is computed by Marian.
He did not generate a sentence when the original is not valid.
- Marian didn't run the validation on the generated sentences, hence the `?`.
- Vlad generated an arbitrary number of fake sentences.
- `log-loss` is computed by Eli using the sentence loss of GPT2 from HuggingFace.
"""


def format_text(text):
    # return text.lower().capitalize() + "."
    return text.lower()


def format_is_valid(is_valid):
    if isinstance(is_valid, str):
        return is_valid
    elif is_valid:
        return "✓"
    else:
        return "×"


def format_log_loss(series, reference):
    if series["type"] == "real":
        color = "black"
    elif series["log-loss"] <= reference:
        color = "green"
    else:
        color = "red"
    log_loss = series["log-loss"]
    return f'<span style="color: {color}">{log_loss:.3f}</span>'


def highlight_difference(sent1, sent2):
    def highlight(word):
        # return f"**{word}**"
        return f'<span style="font-weight: bold; color: red">{word}</span>'

    words1 = sent1.split()
    words2 = sent2.split()
    # try:
    #     assert len(words1) == len(words2)
    # except:
    #     pdb.set_trace()
    return " ".join(
        highlight(word1) if word1 != word2 else word1
        for word1, word2 in zip(words1, words2)
    )


def load_data():
    base_path = Path("output/fake-text")
    path = base_path / "input.txt"
    inp_sents = read_file(str(path), lambda x: x.strip())

    path = base_path / "output-vlad.json"
    with open(path, "r") as f:
        out_sents_vlad = json.load(f)

    path = base_path / "output-vlad-is-valid-sent.json"
    with open(path, "r") as f:
        out_is_valid_vlad = json.load(f)

    path = base_path / "scores-vlad-v2.json"
    with open(path, "r") as f:
        scores_vlad = json.load(f)

    path = base_path / "output-marian.txt"
    out_sents_marian = read_file(str(path), lambda x: x.strip())

    path = base_path / "output-marian-is-valid-sent.txt"
    out_is_valid_marian = read_file(str(path), lambda x: x.split(":")[0] == "TRUE")

    path = base_path / "scores-marian-v2.json"
    with open(path, "r") as f:
        scores_marian = json.load(f)

    def get_score_inp(inp_sent):
        try:
            return scores_vlad[inp_sent]["score0"]
        except KeyError:
            return None

    def get_score_vlad(inp_sent, i, s):
        entry = scores_vlad[inp_sent]["modify"][i]
        sent, score = entry
        assert sent == s
        return score

    def get_is_valid_vlad(inp_sent, i, s):
        if not inp_sent in out_is_valid_vlad:
            return None
        else:
            sent, is_valid, _ = out_is_valid_vlad[inp_sent][i]
            if sent != s:
                return None
            else:
                return is_valid

    def prepare_data_1(i):
        inp_sent = inp_sents[i]
        is_valid_marian = out_is_valid_marian[i]
        datum_real = {
            "type": "real",
            "is-valid": is_valid_marian,
            "log-loss": get_score_inp(inp_sent),
            "text": inp_sent,
        }
        data_fake_vlad = [
            {
                "type": "fake/vlad",
                "is-valid": get_is_valid_vlad(inp_sent, i, s),
                "log-loss": get_score_vlad(inp_sent, i, s),
                "text": highlight_difference(s, inp_sent),
            }
            for i, (s, _, _) in enumerate(out_sents_vlad.get(inp_sent, []))
        ]
        if not is_valid_marian:
            data_fake_marian = []
        else:
            out_sent_marian = out_sents_marian[i]
            data_fake_marian = [
                {
                    "type": "fake/marian",
                    "is-valid": "?",
                    "log-loss": scores_marian[out_sent_marian],
                    "text": highlight_difference(out_sent_marian, inp_sent),
                }
            ]

        return [datum_real] + data_fake_vlad + data_fake_marian

    data = [prepare_data_1(i) for i in range(len(inp_sents))]
    return data


def main():
    with st.sidebar:
        st.markdown(HELP)
        st.markdown("---")
        show_option = st.selectbox(
            "Show",
            options=[
                "random",
                "sort by real's loss (increasing)",
                "sort by real's loss (decreasing)",
            ],
        )
        num = st.number_input(
            "Number of samples",
            min_value=1,
            max_value=300,
            value=30,
            step=10,
        )

    data = load_data()

    # fake_is_better = [
    #     data1[0]["log-loss"] > data1[-1]["log-loss"]
    #     for data1 in data
    #     if data1[0]["log-loss"] is not None
    #     and data1[-1]["log-loss"] is not None
    # ]
    # import numpy as np
    # st.write(len(fake_is_better))
    # st.write(np.sum(fake_is_better))
    # st.write(np.mean(fake_is_better))

    if show_option == "random":
        random.shuffle(data)
    else:
        data = [data1 for data1 in data if data1[0]["log-loss"] is not None]
        to_reverse = show_option == "sort by real's loss (decreasing)"
        data = sorted(data, key=lambda x: x[0]["log-loss"], reverse=to_reverse)

    for data1 in data[:num]:
        df = pd.DataFrame(data1)
        df["text"] = df["text"].apply(format_text)
        df["is-valid"] = df["is-valid"].apply(format_is_valid)
        df["log-loss"] = df[["type", "log-loss"]].apply(
            lambda log_loss: format_log_loss(log_loss, df["log-loss"].iloc[0]), axis=1
        )
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
        st.markdown("---")


if __name__ == "__main__":
    main()
