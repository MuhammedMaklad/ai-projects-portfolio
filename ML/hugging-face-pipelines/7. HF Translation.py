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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="wI6F0nyFVOy2" executionInfo={"status": "ok", "timestamp": 1636891646197, "user_tz": 0, "elapsed": 212187, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}} outputId="191110d3-3a17-4860-c548-bf6da0be8792"
pip install transformers==4.3.0 sentencepiece==0.1.95 nltk==3.5 protobuf==3.15.3 torch==1.7.1

# %% [markdown] id="bOyPfItxWn8t"
# Restart Kernel

# %% id="K8sVtU_YTB9l" colab={"base_uri": "https://localhost:8080/", "height": 176, "referenced_widgets": ["daad464fa9474ecbb457cda3be330a45", "aedbc29d02ab44c99b52767e257c228e", "ed67f7d7e9704316861f046fb2a23c4c", "f53f46816f714a1b910db9e9da74c61a", "6bb4ebfc728240888ee3a5b3a2621329", "af74cd7e02554326a06b052095e3d286", "a752073b76f84d369633f125311909e4", "e04c1478a5c34ce4822fdbac97f7d2d8", "29f4bcaaa9f74ce8876bce070722439a", "1ace3d79ec7c478e979b34f7abfc4090", "e34762300e5f48b784e52964233ac500", "9edd45f5a48f4cbbacfdc4075c2d67c6", "05d07bce67544feeb9774d57653817db", "04f32c822d5048a580a938da1eedf51d", "15c763cde28b42ad836bce08f9e3c1b1", "dd0ade70f1df41edab1ac162a79c4b2f", "bc29f42450da4f489ea2af0efde3b0fd", "1c7e608a297b4b2cbe19f0828daf38bf", "1d4f1d0edfc545688287c2e6f8a79efd", "0f8194135433449786ae0a6a69f3d3e4", "659462c48924469c90ada1fd082952e5", "0a000e743a0c4326b688976b32619585", "a0196a9b436644c79ff011821e309152", "9adfbf0a4b134d41a8f63361532e1262", "28744dd5cdc24e96a834953a3701fab4", "b27357f8ee394e70a80fb50f5a3b015b", "a0a7b0297ba04abe993c2b53e36cd0c6", "3c10352dfe4b45e98f639226bdc591f7", "bd2c23796c85488196c8197968bcbadd", "c3361a908bae43bdbd010b0e9ebb5c1f", "03fb02c76b184b27a3931d1f93a9d8bb", "5b6148d621c14e39b2cf6c1b0fd7e606", "9ac714025d43455885b5a79a474c8676", "9319719e195d49e6b004a729e22ea364", "109d33e358194118ba17241b50effdc1", "0ce58358e4fc477394fdd9508a0c21d5", "6f3b08423f0d4d6d8ad4b66b3e678a00", "eb7de257ae254c918fa5c67decab12de", "bac2269a9547433a8a35a426caf3ae3f", "6c5ee10ae8c9444eafac8b0f9bc1619a", "cd724ee57b0a4ac7a27b74ce5daf0118", "e3ccb8a7f74544c3ad2e88a8fbc0750c", "10cdfc0786204965b7a067dc6d03a463", "69ba7072c4184d24bb5e06c55eb70efc", "30087ab17443471a92f5fbb30e15a96b", "361fa1b04103421a8ab631133e2e95d2", "1165e4265d4f42e18648727c5903af9d", "758b891fe08044b1a4e535062422ebe8", "720beacaec734e63a855ccb873c0a237", "cbac577421124b4eb852e1cd90af8fa0", "d353d468cff7425abe7a378cf5b24406", "5960b09a17a5455583b6dea4c3bbedd5", "f3340e8d9d8643fb99462e1aaa70e4d8", "9f5fc836d5cb4939aed958c83b4237d9", "204bcae75d6445159b4fb5d39112b4bf"]} executionInfo={"status": "ok", "timestamp": 1636891749918, "user_tz": 0, "elapsed": 11405, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}} outputId="8e081906-01f0-4935-c375-0691aabf693f"
from transformers import MarianTokenizer, MarianMTModel
mname = "marefa-nlp/marefa-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(mname)
model = MarianMTModel.from_pretrained(mname)

# %% colab={"base_uri": "https://localhost:8080/"} id="OGg0pUDeUyAt" executionInfo={"status": "ok", "timestamp": 1636891749923, "user_tz": 0, "elapsed": 22, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}} outputId="3e08abc5-d622-49aa-9e90-5c849914047d"
type(tokenizer)

# %% colab={"base_uri": "https://localhost:8080/"} id="K6Te92KnUOfb" executionInfo={"status": "ok", "timestamp": 1636891769892, "user_tz": 0, "elapsed": 5732, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}} outputId="a600000c-a3c6-465a-f8e3-cd983021fb1b"
input = "President Putin went to the presidential palace in the capital, Kiev"

translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([input], return_tensors="pt")) # preparing  , return ids
translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]  # convert id to text

translated_text


# %% id="l3cL3PJiUq29"
def Translate(Text) :
  translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([Text], return_tensors="pt"))
  return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]



# %% colab={"base_uri": "https://localhost:8080/"} id="N6PL9GJiW1WQ" executionInfo={"status": "ok", "timestamp": 1636891926351, "user_tz": 0, "elapsed": 38630, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}} outputId="ad953aaa-b2d8-41e9-e361-9a575b4108c8"
Text = '''
Joseph Vissarionovich Stalin[f] (18 December [O.S. 6 December] 1878[1] – 5 March 1953) was a Georgian revolutionary and Soviet political leader who governed the Soviet Union from 1924 until his death in 1953. He served as both General Secretary of the Communist
Party of the Soviet Union (1922–1952) and Chairman of the Council of Ministers of the Soviet Union (1941–1953). Despite initially governing the country as part of a collective leadership, he ultimately consolidated power to become the Soviet Union's dictator by the 1930s.
 A communist ideologically committed to the Leninist interpretation of Marxism, Stalin formalised these ideas as Marxism–Leninism while his own policies are known as Stalinism.

Born to a poor family in Gori in the Russian Empire (now Georgia), Stalin attended the Tbilisi Spiritual Seminary before eventually joining the Marxist Russian Social Democratic Labour Party. He went on to edit the party's newspaper, Pravda, and raised funds for
Vladimir Lenin's Bolshevik faction via robberies, kidnappings and protection rackets. Repeatedly arrested, he underwent several internal exiles. After the Bolsheviks seized power during the October Revolution and created a one-party state under the newly formed Communist
Party in 1917, Stalin joined its governing Politburo. Serving in the Russian Civil War before overseeing the Soviet Union's establishment in 1922, Stalin assumed leadership over the country following Lenin's death in 1924. Under Stalin, socialism in one country became a central
tenet of the party's dogma. As a result of the Five-Year Plans implemented under his leadership, the country underwent agricultural collectivisation and rapid industrialisation, creating a centralised command economy. This led to severe disruptions of food production that contributed
to the famine of 1932–33. To eradicate accused "enemies of the working class", Stalin instituted the Great Purge, in which over a million were imprisoned and at least 700,000 executed between 1934 and 1939. By 1937, he had absolute control over the party and government.
'''
Translate(Text)
