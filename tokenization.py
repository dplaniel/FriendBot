# import parse_text
import pandas as pd
import string
import numpy as np
from transformers import TFGPT2LMHeadModel
from transformers import GPT2TokenizerFast

N_MOST_COMMON = 1750

df = pd.read_pickle("./this_is_the_text_chat.pkl")  # this shouldn't be hardcoded huh


messages = df["clean_content"]

words = []
for msg in messages:
    tokerinos = msg.split()
    words += tokerinos

unq_words = {}
for word in words:
    unq_words[word] = unq_words.get(word, 0) + 1
words_by_ct = sorted(unq_words.keys(), key=lambda x: unq_words[x], reverse=True)
word_list = [word for word in words_by_ct[:N_MOST_COMMON]]

with open("emojis_list.csv", "r") as infile:  # nor should this be hardcoded
    emojis = [line.strip() for line in infile.readlines()]


friendbot_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

print(f"Starting vocab size: {len(friendbot_tokenizer)}")
friendbot_tokenizer.add_tokens(emojis)

novel_tokens = set()
for i in range(0, N_MOST_COMMON, 64):
    word_subset = word_list[i : i + 64]
    try:
        n = friendbot_tokenizer.add_tokens(word_subset)
    except Exception:
        break
    # print(n)


print(f"Ending vocab size: {len(friendbot_tokenizer)}")

friendbot_tokenizer.save_pretrained("friendbot_tokenizer_extended_vocab.chkpt")
