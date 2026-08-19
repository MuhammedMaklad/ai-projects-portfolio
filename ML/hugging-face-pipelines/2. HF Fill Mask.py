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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3812, "status": "ok", "timestamp": 1636887455712, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="j2tLXI3TzqJl" outputId="a46ea39d-ab12-47b8-9a1e-3b0d791fc565"
pip install transformers

# %% [markdown] id="xZXGdJJ7ngLO"
# # 'albert-base-v2' for fill-mask Task

# %% id="MyqcuhblzmWA"
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")  # complete empty wor, ids

model = AutoModelForMaskedLM.from_pretrained("albert-base-v2") # model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1749, "status": "ok", "timestamp": 1636887462923, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="-sFbAjSzz5tn" outputId="c3ca7545-6c22-4f09-c5e5-e22aad1febf9"
from transformers import pipeline
unmasker = pipeline('fill-mask', model='albert-base-v2')
results = unmasker("The man worked as a [MASK].")
results

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1636887462925, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="IKOs_pUCz5rA" outputId="1de36b51-a422-42d1-e23a-e54ae7f95823"
for r in results :
  print(f'result : ({r["sequence"]})  with score : ({r["score"]}) ')


# %% id="0gGDnnRsz5n8"
def CompletePhrase(text) :
  unmasker = pipeline('fill-mask', model='albert-base-v2')
  results = unmasker(text)
  for r in results :
    print(f'result : ({r["sequence"]})  with score : ({r["score"]}) ')


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1636887462928, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="4E8qO6tczmS4" outputId="0bfd2ed1-3ed6-43a8-8169-f817d238f08a"
CompletePhrase('[MASK] is the capital of Greece')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2487, "status": "ok", "timestamp": 1636887465397, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="fBhUKRPezmOt" outputId="c675007f-0bbd-4dd2-c6f8-a61437e97920"
CompletePhrase('I called [MASK] today')

# %% [markdown] id="CqeIUq3jngLQ"
# # 'xlm-roberta-base' for fill-mask Task

# %% colab={"base_uri": "https://localhost:8080/", "height": 437, "referenced_widgets": ["9faeef9553b54d8684ebb556b1ccd9ea", "d82e4fb5b8384606bea7f37bdd21bc35", "01a62d55af2b4dca833252d1e575b2fa", "a11f6ca3273e4009b368b8d639068152", "694b9696b9dd449cad29b762e0522130", "a859daa78be946b7bab0e03b26ed4d77", "dd423643bef6417aa1cc1351011e1e40", "efad42d7280e4644b36b3ee4a69336cb", "b8890cb7584e42ebb93b83969843f474", "a5ab7ad6711447bd8163e0f57c367095", "3c1ea8cd89da438aa7106fd1620579fc", "b5036a1e53f543e5ba5e1cc30feed6ac", "a832c4e14e60423b9edb8b4a01e934ec", "397048838bdd47c99db26bd7ce676081", "261d8a1c8d4d46ab9c77e2bb1da0d785", "564a20d5e33a413e81115921a97ffcbc", "8a111605b0304b0384e5de2e57289127", "f0a5590a7b584bee9d6f48a77ee736fb", "50ee3ae4753c44059ccc6c1c3a05039a", "db7f0aa82d5c41878cf8f85e74566dfc", "854aae9eed304b5abc9b1293f15ef4c2", "4462905533394edf91efdfb4b9196fb6", "8eba5cb93c2a4f8da24d670b1872ab79", "0d893669e3e84ee3963f50e85acce862", "d8292be982b54a6bac73eb46fdd4a98c", "ac0f1e01186e4b5e994fdca41562ce06", "a774c2ddaa04417399f4c70d1f860b2b", "32f60d26f68b4b31b6cc7cf47c25e8b4", "2c07eafaf5ae47e6b0d9092b3a701e1f", "6025d1aa2d0e4603859c6d232803ddf9", "f49275687d834c2fb757b95f78407e26", "e30b2944208946d1b8be661bd702e78a", "84f33eef9ab147d3b9af5fb5fed9b263"]} executionInfo={"elapsed": 33489, "status": "ok", "timestamp": 1636887498831, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="Py3rruxaF7Vg" outputId="9144931f-6c37-4555-c910-8dca37403774"
unmasker = pipeline('fill-mask', model='xlm-roberta-base')
unmasker("Hello I'm a <mask> model.")


# %% id="GMYqPYBPF-YE"
def CompletePhrase(text) :
  unmasker = pipeline('fill-mask', model='xlm-roberta-base')
  results = unmasker(text)
  for r in results :
    print(f'result : ({r["sequence"]})  with score : ({r["score"]}) ')


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7166, "status": "ok", "timestamp": 1636887597375, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="9wMxPoZjGXwx" outputId="896dddb7-2a83-4963-90f5-0107999305a3"
CompletePhrase('<mask> is the capital of Greece')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6186, "status": "ok", "timestamp": 1636887622665, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="udFiK1ybGXwz" outputId="a352646a-6db0-40c0-8a4b-4840a4b56463"
CompletePhrase('I called <mask> today')
