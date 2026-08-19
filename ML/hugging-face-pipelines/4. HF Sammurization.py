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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9459, "status": "ok", "timestamp": 1636889253534, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="j2tLXI3TzqJl" outputId="d921d24b-709f-470b-c98e-40a6a40a6adc"
pip install transformers

# %% id="YynGvaWAOmiy"
from transformers import pipeline
summarizer = pipeline(task = "summarization", model="philschmid/bart-large-cnn-samsum")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18364, "status": "ok", "timestamp": 1636889761872, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="f_LYmZ7hM5HL" outputId="87b98432-21f0-438d-9460-0809dbf22fd6"
article_text = '''
Joseph Vissarionovich Stalin[f] (18 December [O.S. 6 December] 1878[1] – 5 March 1953) was a Georgian revolutionary and Soviet political leader who governed the Soviet Union from 1924 until his death in 1953. He served as both General Secretary of the Communist
Party of the Soviet Union (1922–1952) and Chairman of the Council of Ministers of the Soviet Union (1941–1953). Despite initially governing the country as part of a collective leadership, he ultimately consolidated power to become the Soviet Union's dictator by the 1930s.
 A communist ideologically committed to the Leninist interpretation of Marxism, Stalin formalised these ideas as Marxism–Leninism while his own policies are known as Stalinism.

Born to a poor family in Gori in the Russian Empire (now Georgia), Stalin attended the Tbilisi Spiritual Seminary before eventually joining the Marxist Russian Social Democratic Labour Party. He went on to edit the party's newspaper, Pravda, and raised funds for
Vladimir Lenin's Bolshevik faction via robberies, kidnappings and protection rackets. Repeatedly arrested, he underwent several internal exiles. After the Bolsheviks seized power during the October Revolution and created a one-party state under the newly formed Communist
Party in 1917, Stalin joined its governing Politburo. Serving in the Russian Civil War before overseeing the Soviet Union's establishment in 1922, Stalin assumed leadership over the country following Lenin's death in 1924. Under Stalin, socialism in one country became a central
tenet of the party's dogma. As a result of the Five-Year Plans implemented under his leadership, the country underwent agricultural collectivisation and rapid industrialisation, creating a centralised command economy. This led to severe disruptions of food production that contributed
to the famine of 1932–33. To eradicate accused "enemies of the working class", Stalin instituted the Great Purge, in which over a million were imprisoned and at least 700,000 executed between 1934 and 1939. By 1937, he had absolute control over the party and government.
'''


summarizer(article_text)[0]['summary_text']
