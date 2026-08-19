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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9170, "status": "ok", "timestamp": 1636887993657, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="j2tLXI3TzqJl" outputId="e1d641d2-c13c-455b-d865-d5465587a303"
pip install transformers

# %% [markdown]
# # "deepset/roberta-base-squad2" for 'question-answering' Task

# %% colab={"base_uri": "https://localhost:8080/", "height": 208, "referenced_widgets": ["c7e109bd790d44f8a7edd09cffab833d", "ff44c04ec6bd41c486531634323bb03f", "a7de7af37f064ff8bb2503509803c1c2", "a6056cfc22b34b3f92594fad35aa4ea6", "0a77dc95822a4ab4a6c95887cb98ce78", "2a01bc470f284151b3c909ac2e898422", "3128a4ab31a44fcd81a0834f52de9267", "86fdbe6d15ff45f59cb497d10789bc99", "39c7bace95a24bbe8e887448da5b7ed3", "35321b93614c4927801154d0aadd3241", "4f5aaad52d4a4624b5391807d08691df", "9f25094e327c43a79dd64ee0336e0478", "602307afce94477884a7c3d5738687c1", "0385cf875141470ca186e4503a009f8d", "aab8e95afafe4aada333f74ce98418c5", "bcb17728c95840e7a18173c427d3da46", "b0c2ef49192b4802bcc42e64bb8b4285", "2b18d6b2645c4231aa129f39b69b394d", "4c6ed2ebae9b4321a5cb34edd96b24f4", "b9139528bb0d465abf9afd68d683ff54", "0af7377283d5492b939371b2b57e98f8", "3a067094f3a54b8bb7973d0da7ec1cde", "79afd023c8c541eba122223bdf340737", "fed6ff9b1c904d75a6327731c0e81572", "2991228c362b4a679d5eecce0e794183", "da7af71e3d094037adef7d9b192db64d", "66a4eeaf5322450eaeaf83ee33c8b2eb", "2b8b27c0f8fc40178083907f43c9842d", "d317a9ee294541c4be5585ed47281106", "020a5385c2a44a05b0d858f1928cb6e1", "43520375b9154d85af82deed611ef3b2", "9978c12412fd44979a4185d1b81748d3", "87df6d0cb3fc44929ef8ba6bfa44f287", "4c30711dae544f8aad8c82ccd1726705", "72f261e1ef464e69b4925e68fccdbe96", "6397d5b01d9d4898b81de8cacfe18d31", "d007e8cfb0504154b33ce3f56499b329", "1395f44e756a4d01a7e28dde0ac7704d", "3b04f22008834ab1917975b0814f71db", "9dbe3843fb8a4ebea96b67d950efb9f4", "b4f8ca7e0f4347509c6b4e10f50d0c11", "639a6d5e4a6c43ba90ec1b81ec63ea52", "ede25d398e0c4bffb35e037ea4b659cc", "d5dda5221a8c4765a84acd2d9e9ab3e8", "cff18274c91847b9abd84e310230e67c", "f3334569b2284480b207f8ab75e1ac59", "500809f63dcb4c7fa3c1adb6efb80d7d", "88b765f80e9f48b48014a1ec3b7c1075", "959a3d1e4fa042ed95a2e232e2762cec", "130bfdf243e343c78329b0fc6f36efdf", "b7cd84a9e9a94697b8ba69a5db58daa8", "e3840b2dcfd54b4d86743b8c1241ee33", "c4f5c2eb1c1544f79c71a17ddfb8c92b", "532ca7782cdc43e59fd21dbb2ccac9e3", "15fee95623de4d8b92c98c7d4ead1747", "cdd669d1822145a49a56c006496d4918", "cfcc4a5d73c3432388d2e6a18180622e", "47a832f305cd4c8797f87b9500b7cc6f", "f3bf4eb086ab4f90bb953d257634cb2c", "478e2b142899427eb04da5969137c1fc", "328367fae44c4c409d9cf8dda38bd245", "1cefde0e0a9545f681ef3594b44e1a5d", "f8d7720332204d2098380b65d6437f83", "78a6449159524652880b269191d14055", "5f1b90b190a94772a9629a5351ba6a03", "cbdea454b6a14dd6bcc1194e54874011"]} executionInfo={"elapsed": 48578, "status": "ok", "timestamp": 1636888043453, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="MyqcuhblzmWA" outputId="f7f7d4c1-f8d1-47c2-f8dd-31a30dd178d7"
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2340, "status": "ok", "timestamp": 1636888124607, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="8Oj6uUCjIhaJ" outputId="a1a1db98-6149-41b3-e9f1-5cd28f2b7e4e"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
nlp(QA_input)


# %% executionInfo={"elapsed": 252, "status": "ok", "timestamp": 1636888215581, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="E2G0DT79ImH3"
def AnsweringQuestions(Question,Context) : 
  model_name = "deepset/roberta-base-squad2"
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = {'question': Question,'context': Context}
  return nlp(QA_input)['answer']


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 2242, "status": "ok", "timestamp": 1636888253238, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="w3a834fSImFK" outputId="1375e28f-8b9d-4c17-d0d7-33dd86b74e87"
AnsweringQuestions('who is Ahmed ? ', 'I met my cousin Ahmed who works as data scientist')

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 2107, "status": "ok", "timestamp": 1636888282004, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="wBgVZDqnImCm" outputId="64ea1641-d3bc-4d93-ef2f-b6f97ceb39d1"
AnsweringQuestions('what is Ahmed`s job ? ', 'I met my cousin Ahmed who works as data scientist')

# %% colab={"base_uri": "https://localhost:8080/", "height": 87} executionInfo={"elapsed": 7504, "status": "ok", "timestamp": 1636888574383, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="5Fo00ZyFIl81" outputId="630409e4-2c80-403b-e845-0b36903bc85a"
WW2 = '''
World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all of the great powers—forming two opposing military alliances: the Allies and the Axis powers. In a total war directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. Aircraft played a major role in the conflict, enabling the strategic bombing of population centres and the only two uses of nuclear weapons in war to this day. World War II was by far the deadliest conflict in human history; it resulted in 70 to 85 million fatalities, a majority being civilians. Tens of millions of people died due to genocides (including the Holocaust), starvation, massacres, and disease. In the wake of the Axis defeat, Germany and Japan were occupied, and war crimes tribunals were conducted against German and Japanese leaders.

World War II is generally considered to have begun on 1 September 1939, when Nazi Germany, under Adolf Hitler, invaded Poland. The United Kingdom and France subsequently declared war on Germany on 3 September. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union had partitioned Poland and marked out their "spheres of influence" across Finland, Romania and the Baltic states. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan (along with other countries later on). Following the onset of campaigns in North Africa and East Africa, and the fall of France in mid-1940, the war continued primarily between the European Axis powers and the British Empire, with war in the Balkans, the aerial Battle of Britain, the Blitz of the UK, and the Battle of the Atlantic. On 22 June 1941, Germany led the European Axis powers in an invasion of the Soviet Union, opening the Eastern Front, the largest land theatre of war in history.

Japan, which aimed to dominate Asia and the Pacific, was at war with the Republic of China by 1937. In December 1941, Japan attacked American and British territories with near-simultaneous offensives against Southeast Asia and the Central Pacific, including an attack on the US fleet at Pearl Harbor which forced the US to declare war against Japan; the European Axis powers declared war on the US in solidarity. Japan soon captured much of the western Pacific, but its advances were halted in 1942 after losing the critical Battle of Midway; later, Germany and Italy were defeated in North Africa and at Stalingrad in the Soviet Union. Key setbacks in 1943—including a series of German defeats on the Eastern Front, the Allied invasions of Sicily and the Italian mainland, and Allied offensives in the Pacific—cost the Axis powers their initiative and forced it into strategic retreat on all fronts. In 1944, the Western Allies invaded German-occupied France, while the Soviet Union regained its territorial losses and turned towards Germany and its allies. During 1944 and 1945, Japan suffered reversals in mainland Asia, while the Allies crippled the Japanese Navy and captured key western Pacific islands.

The war in Europe concluded with the liberation of German-occupied territories, and the invasion of Germany by the Western Allies and the Soviet Union, culminating in the fall of Berlin to Soviet troops, Hitler's suicide and the German unconditional surrender on 8 May 1945. Following the Potsdam Declaration by the Allies on 26 July 1945 and the refusal of Japan to surrender on its terms, the United States dropped the first atomic bombs on the Japanese cities of Hiroshima, on 6 August, and Nagasaki, on 9 August. Faced with an imminent invasion of the Japanese archipelago, the possibility of additional atomic bombings, and the Soviet entry into the war against Japan and its invasion of Manchuria, Japan announced its intention to surrender on 15 August, then signed the surrender document on 2 September 1945, cementing total victory in Asia for the Allies.

World War II changed the political alignment and social structure of the globe. The United Nations (UN) was established to foster international co-operation and prevent future conflicts, and the victorious great powers—China, France, the Soviet Union, the United Kingdom, and the United States—became the permanent members of its Security Council. The Soviet Union and the United States emerged as rival superpowers, setting the stage for the nearly half-century-long Cold War. In the wake of European devastation, the influence of its great powers waned, triggering the decolonisation of Africa and Asia. Most countries whose industries had been damaged moved towards economic recovery and expansion. Political integration, especially in Europe, began as an effort to forestall future hostilities, end pre-war enmities and forge a sense of common identity.

'''

AnsweringQuestions('when war began? ', WW2)

# %% [markdown]
# # "deepset/electra-base-squad2" for 'question-answering' Task

# %% colab={"base_uri": "https://localhost:8080/", "height": 241, "referenced_widgets": ["ad7177153d464e1abe7b4a54ae0626a6", "7c61ce7c113d4b92ac7c0f56f35ef5c6", "6e6f0d49cff04d99881c19f4ef88519f", "1cec2b0732164895afab2f8393c54e85", "417b203dbaa14e9c8e5630967484f2ae", "28b47794f9fa4491b792927d139a019d", "0e69bda8f58b447a86147f0cc78593a7", "f4db41694eec43139e18ed59742fc938", "7c7e3662685a4b638c2ce418ba6a76d8", "00d647ed60bd4c9295d7d5dbc2038d5b", "d9fbfeaa7d524455b146fc17f28beae0", "6e6a92a536c64f9a8636204470abfa14", "e73419a4591d4c788bd7f4a0b3aad743", "78a918bc94994fe783b13598cc95e5cf", "404db9b0c7e345a98ac03e99f529306e", "fc655d4b06394d45be22e33685377f2e", "87c49a0d584e4653a082fc33c3879d16", "9577e1e01d984247816f595842fc4029", "7fde1833f9924cdc970a0c39f12166c1", "543d05a36c3548fbabf36d4b04e54ca0", "2a084bd11ea5475c8beef286f12c4652", "f12b9aab6877432cbacf5a158cf6d40c", "5b4aeaa1329f4f7e9e1746ef77645f3e", "815b69ace421444084aa1d97a6c06553", "f651c287cf56488484f0bc6e3828d537", "47fb10a7b85d44e5af456d66dbc09d4d", "8c9ce303df7e4d91a74468b8347c6553", "adf78d61959e40a3935c72c21bea8503", "a89a982637314dccbda8db324b589133", "d0fec5eee1a54e12811e9ab880305159", "4ef20cc5d9324a38b3d5101119c969a8", "e4c53069570b4dc3a441eb0b764d5a2b", "db58e310150147b1aecedba72e13a265", "3b83e5ac0f964f7bbc95b05bea51baeb", "3d5204fd6f584956b417901bb7302e6d", "1aa7843f65384c91af4c47de7702eeaa", "88257f1f0a2e469c85a0d3c502c877a7", "304ad0d05e3c4d218555c15346ad8a62", "7bb34804841843718b8b342306aaca37", "1d66adfb2cd648c5ba2ad7e7dfe0bd29", "796af1fd113b45eea8e53c180bc4a4f2", "10017b8f9a0a473c9c35977fc4c9c065", "31235de7dde64e258b483b6b17bf92d5", "a5e6202d0b284d4092d5a8618b8a40b8", "341df4d513694ef4b977685b9fe675b2", "eb1d50481af64b2c91e0c8c2f67cece9", "f7fac5ac63d346f788f8d3a617604afe", "1eb2bbb68e8241a4965110ada4dd2f19", "a9d4e5d3dcda4996ad056876f95e5453", "2e970c35b09e4ab3af9a1e7565c7662c", "ee6ac13e79604a668ee52e6c225bbbbf", "c49850e49a884f6797a59e36f6c13c99", "9fa12b9f65064311a69fc4ec17fc226f", "94042fb4454f407b865cfeceb388650e", "eadb317fdc5a4cd4a6fc3f187d068ce8"]} executionInfo={"elapsed": 15124, "status": "ok", "timestamp": 1636888744535, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="3xKZkxxvIhXe" outputId="6be9f259-a87b-48cb-b96b-a35bafe877b0"
model_name = "deepset/electra-base-squad2"


nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
nlp(QA_input)


# %% executionInfo={"elapsed": 319, "status": "ok", "timestamp": 1636888790180, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="fbD-z0oQLAza"
def AnsweringQuestions(Question,Context) : 
  model_name = "deepset/electra-base-squad2"
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = {'question': Question,'context': Context}
  return nlp(QA_input)['answer']


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 1821, "status": "ok", "timestamp": 1636888793233, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="qx19ff7-LAzb" outputId="14949c21-febd-4578-8e2f-7ca15e6e0770"
AnsweringQuestions('who is Ahmed ? ', 'I met my cousin Ahmed who works as data scientist')

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 1569, "status": "ok", "timestamp": 1636888795055, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="LUo1eIF1LAzc" outputId="01896daf-ab16-4e49-a47a-a07ba71405d4"
AnsweringQuestions('what is Ahmed`s job ? ', 'I met my cousin Ahmed who works as data scientist')

# %% colab={"base_uri": "https://localhost:8080/", "height": 87} executionInfo={"elapsed": 6879, "status": "ok", "timestamp": 1636888807227, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="n49RpVEXLAze" outputId="b47f7074-cbaa-4d82-e113-2592456fce5a"
WW2 = '''
World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all of the great powers—forming two opposing military alliances: the Allies and the Axis powers. In a total war directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. Aircraft played a major role in the conflict, enabling the strategic bombing of population centres and the only two uses of nuclear weapons in war to this day. World War II was by far the deadliest conflict in human history; it resulted in 70 to 85 million fatalities, a majority being civilians. Tens of millions of people died due to genocides (including the Holocaust), starvation, massacres, and disease. In the wake of the Axis defeat, Germany and Japan were occupied, and war crimes tribunals were conducted against German and Japanese leaders.

World War II is generally considered to have begun on 1 September 1939, when Nazi Germany, under Adolf Hitler, invaded Poland. The United Kingdom and France subsequently declared war on Germany on 3 September. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union had partitioned Poland and marked out their "spheres of influence" across Finland, Romania and the Baltic states. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan (along with other countries later on). Following the onset of campaigns in North Africa and East Africa, and the fall of France in mid-1940, the war continued primarily between the European Axis powers and the British Empire, with war in the Balkans, the aerial Battle of Britain, the Blitz of the UK, and the Battle of the Atlantic. On 22 June 1941, Germany led the European Axis powers in an invasion of the Soviet Union, opening the Eastern Front, the largest land theatre of war in history.

Japan, which aimed to dominate Asia and the Pacific, was at war with the Republic of China by 1937. In December 1941, Japan attacked American and British territories with near-simultaneous offensives against Southeast Asia and the Central Pacific, including an attack on the US fleet at Pearl Harbor which forced the US to declare war against Japan; the European Axis powers declared war on the US in solidarity. Japan soon captured much of the western Pacific, but its advances were halted in 1942 after losing the critical Battle of Midway; later, Germany and Italy were defeated in North Africa and at Stalingrad in the Soviet Union. Key setbacks in 1943—including a series of German defeats on the Eastern Front, the Allied invasions of Sicily and the Italian mainland, and Allied offensives in the Pacific—cost the Axis powers their initiative and forced it into strategic retreat on all fronts. In 1944, the Western Allies invaded German-occupied France, while the Soviet Union regained its territorial losses and turned towards Germany and its allies. During 1944 and 1945, Japan suffered reversals in mainland Asia, while the Allies crippled the Japanese Navy and captured key western Pacific islands.

The war in Europe concluded with the liberation of German-occupied territories, and the invasion of Germany by the Western Allies and the Soviet Union, culminating in the fall of Berlin to Soviet troops, Hitler's suicide and the German unconditional surrender on 8 May 1945. Following the Potsdam Declaration by the Allies on 26 July 1945 and the refusal of Japan to surrender on its terms, the United States dropped the first atomic bombs on the Japanese cities of Hiroshima, on 6 August, and Nagasaki, on 9 August. Faced with an imminent invasion of the Japanese archipelago, the possibility of additional atomic bombings, and the Soviet entry into the war against Japan and its invasion of Manchuria, Japan announced its intention to surrender on 15 August, then signed the surrender document on 2 September 1945, cementing total victory in Asia for the Allies.

World War II changed the political alignment and social structure of the globe. The United Nations (UN) was established to foster international co-operation and prevent future conflicts, and the victorious great powers—China, France, the Soviet Union, the United Kingdom, and the United States—became the permanent members of its Security Council. The Soviet Union and the United States emerged as rival superpowers, setting the stage for the nearly half-century-long Cold War. In the wake of European devastation, the influence of its great powers waned, triggering the decolonisation of Africa and Asia. Most countries whose industries had been damaged moved towards economic recovery and expansion. Political integration, especially in Europe, began as an effort to forestall future hostilities, end pre-war enmities and forge a sense of common identity.

'''

AnsweringQuestions('when war began? ', WW2)

# %% id="4g9urvUvLOpy"
