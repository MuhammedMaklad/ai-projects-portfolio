# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9609, "status": "ok", "timestamp": 1636890681186, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="j2tLXI3TzqJl" outputId="d3eb0c12-1d42-4be5-b2d7-96cb3adc2227"
pip install transformers

# %% colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["513aa701c5dd489dba86a79eb2072e73", "8a51ace84b9246f880aa484f38714517", "8d211638e2384d4b8cae47bfe10719d4", "61f148f1a5514081b080ec5b2831b8ec", "31cc075f33ea4cbfa1a769185a49a303", "f896e1bd1c6940658eb150c3b01e9ee5", "f5b485cd708d43f98cadbcd103d59fb8", "ead6a370d3ec4c66a9471b813e75ad9b", "96f1b3978c804993b5e1be03b7c1b7e6", "02932900ff5e4830b5f7d5b1e1904cb0", "cde9b1f22379492d991066df6b49b9cd", "544e028c660643dab0b9bbe312d93e50", "e579cdba02e04dfdbad2d00fcfb045a2", "fe1707b7e2604cdcbcae6501acb9a14e", "424b75d799914762823e27707558ff2b", "6ebd017ca0584fe2af90f3297d321463", "aa9d2cd03e0c4259abb173ff4b2d00f4", "49c1f9aec3a84d2389af8bf8861e2d87", "b16f00b3d1ee462a82289fc72377b098", "f222f55640644cdc931ae9799418f175", "04148ef3a7a44a0d8a706a20ae60555b", "dad0d75820c342cfa379193ba97c0f1d", "9ce033239878442a9141767d12f5448b", "043296262cf44641982f91877a528a74", "a187f7690cf84d17a3fdc50e41ea3fe1", "fee98fcdf33d42529ac839dc0607f5d0", "7e3d4b3342ce41c2b5872d96be535d45", "9406061fa13441a9b7abb015fac713af", "3ef48fe12ccd4e3e921eb76c10c0b44f", "b7bfba54598e4aa3b613ff9b31502974", "55cd168c08054443be4edc7ee1b7dbea", "03529090ddf8432f8af6c673e895f77a", "74909e58e8b046f78e3bd20a4ff2b1f0", "88c5e65edf57479b96d901842bceef78", "28895d199f334d8985fdce56ca862244", "c8d8ce7867c44f3cb50c10d227bfedda", "35b26975aa6943c2a37e6d54393dbfce", "1d62c0d0c90f41609ee88df9b3ce0ae0", "ccc97cc7b8ef4e8485b15579d7b7cced", "0166b28a9c554e148a8c53304f54a523", "11830dcc49204532a43e3ad061230dfe", "b97bdecdd054421fb16996926ed2417a", "731c2e7388fb4e1c866fe2d6fc0af592", "a97c838279584ac98d01b62e2595bb1c", "531868cf979a4259b22b4dd6c5d8d7ed", "bfa75311221c4c1a960cc474b13be489", "715051da0f9f43dfb1c7c7e6feb3bbab", "8f7bcdf035224054bdb6e6f087b45d8f", "affe1e7ad7c44f9d9bf49b7be03026eb", "8c54bdd3fe7944faba99920160a1db6d", "4f9311a0823140019c66b0b19bd335ea", "42e4ace14b1e45e79d5c5a4b59873053", "44dee48c8b2143bdb2384546ec4a5d9b", "e8fb03772cf3406f86619e4b86da72c8", "faab8277c9a44469a5f28025feec3f9c"]} executionInfo={"elapsed": 55012, "status": "ok", "timestamp": 1636890736186, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="k5R7A2HzRuUd" outputId="f9258bc6-240b-4963-9df9-4e0051b08fca"
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3294, "status": "ok", "timestamp": 1636890787592, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="mwc6usInSUUA" outputId="1ec6ea6d-db94-4f7a-b191-3f966d218cfc"
generator("can I ask you a question", max_length=30, num_return_sequences=5)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3146, "status": "ok", "timestamp": 1636890848541, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="lzFWwIS-Sy9h" outputId="b2938e45-05b9-4ac3-97ff-43eb0ea78b61"
generator("Albert Einstein is one of the most", max_length=30, num_return_sequences=5)
