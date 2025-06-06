import os
discord_channel_id = os.getenv('DISCORD_CHANNEL_ID')
discord_api = f'https://discord.com/api/v10/channels/{discord_channel_id}/messages'
discord_bot_token = os.getenv('DISCORD_BOT_TOKEN')
authorization_header = {'Authorization': f'Bot {discord_bot_token}'}
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

inference_access_token = os.getenv('HF_INFERENCE_ACCESS_TOKEN')
inference_header = {'Authorization': f'Bearer {inference_access_token}'}
inference_api = 'https://api-inference.huggingface.co'


import httpx
httpx.post(discord_api, headers=authorization_header, json={'content': 'lambda_function.py'})
def debug_mode(request_body):
    # # https://developers.line.biz/en/reference/messaging-api/#request-body
    # destination = request_body['destination']
    # httpx.post(discord_api, headers=authorization_header, json={'content': destination})
    events = request_body['events']
    if events == []:
        httpx.post(discord_api, headers=authorization_header, json={'content': 'Webhook URL Verify Success'})
    elif events[0]['type'] == 'follow':
        httpx.post(discord_api, headers=authorization_header, json={'content': f"followed by {events[0]['source']['type']}Id\n" + events[0]['source'][f"{events[0]['source']['type']}Id"]})
    elif events[0]['type'] == 'unfollow':
        httpx.post(discord_api, headers=authorization_header, json={'content': f"unfollowed by {events[0]['source']['type']}Id\n" + events[0]['source'][f"{events[0]['source']['type']}Id"]})
    elif events[0]['type'] == 'message':
        httpx.post(discord_api, headers=authorization_header, json={'content': f"{events[0]['message']['type']} from {events[0]['source']['type']}Id\n" + events[0]['source'][f"{events[0]['source']['type']}Id"]})
    else:
        httpx.post(discord_api, headers=authorization_header, json={'content': f"{events[0]['type']}"})
def god_mode(Q, A):
    Q = f'\n🤔：{Q}'
    A = f'\n🤖：{A}'
    httpx.post(discord_api, headers=authorization_header, json={'content': Q+A})

import re
import base64
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    StickerMessageContent,
    AudioMessageContent,
    ImageMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
    AudioMessage,
    ImageMessage
)
configuration = Configuration(access_token=channel_access_token)
with ApiClient(configuration) as api_client:
    line_bot_api = MessagingApi(api_client)
    line_bot_blob_api = MessagingApiBlob(api_client)
def show_loading_animation(event):
    line_bot_api.show_loading_animation(
        ShowLoadingAnimationRequest(
            chat_id=event.source.user_id,
            # loading_seconds=5
        )
    )
handler = WebhookHandler(channel_secret)
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_text = event.message.text
    if event.source.type != 'user':
        if m := re.search('@(Agent )?PHIL', user_text, flags=re.IGNORECASE):
            user_text = user_text.replace(m.group(), m.group()[1:])
        else:
            return
    show_loading_animation(event)
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=assistant_messages(event, user_text)
        )
    )
@handler.add(MessageEvent, message=StickerMessageContent)
def handle_sticker_message(event):
    if event.source.type != 'user':
        return
    show_loading_animation(event)
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[
                TextMessage(text='$', emojis=[{'index': 0, 'productId': '5ac21c46040ab15980c9b442', 'emojiId': '138'}])
            ]
        )
    )
@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event):
    message_id = event.message.id
    message_content = line_bot_blob_api.get_message_content(message_id=message_id)
    with open(f'/tmp/{message_id}.m4a', 'wb') as tf:
        tf.write(message_content)
    # transcript = openai_client.audio.transcriptions.create(
    #     model='whisper-1',
    #     file=open(f'/tmp/{message_id}.m4a', 'rb'),
    #     response_format='text'
    #     ).strip()
    httpx.post(discord_api, headers=authorization_header, json={'content': model_generates_transcript})
    try:
        # d = httpx.post(f'{inference_api}/models/{model_generates_transcript}', headers=inference_header, json=message_content).json()
        # transcript = d.get('text')
        output = inference_client.automatic_speech_recognition(f'/tmp/{message_id}.m4a', model=model_generates_transcript)
        transcript = output.text
        # error = d.get('error')
        # if error:
        #     httpx.post(discord_api, headers=authorization_header, json={'content': error})
        #     return
    except Exception as e:
        httpx.post(discord_api, headers=authorization_header, json={'content': e})
        return
    show_loading_animation(event)
    messages = assistant_messages(event, transcript)
    # openai_client.audio.speech.create(model='tts-1', voice='onyx', input=messages[-1].text).stream_to_file(f'/tmp/{message_id}.mp3')
    edge_tts.Communicate(messages[-1].text, voice).save_sync(f'/tmp/{message_id}.mp3')
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=messages + [
                AudioMessage(
                    original_content_url=s3_object_url(f'/tmp/{message_id}.mp3'),
                    duration=60000
                )
            ]
        )
    )
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    message_id = event.message.id
    message_content = line_bot_blob_api.get_message_content(message_id=message_id)
    source_id = eval(f'event.source.{event.source.type}_id') # user/group/room
    with open(f'/tmp/{source_id}.jpg', 'wb') as tf:
        tf.write(message_content)
    s3_object_url(f'/tmp/{source_id}.jpg')
    global after_image_message
    after_image_message = True
after_image_message = False


from openai import OpenAI
# openai_client = OpenAI()
# ollama_client = OpenAI(base_url=f"{os.getenv('OLLAMA_HOSTNAME')}/v1", api_key='ollama')
# model_supports_tools = 'llama3.2'
# model_supports_vision = 'llama3.2-vision'
# model_generates_text = 'llama3.2-vision'
xai_client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv('XAI_API_KEY'))
model_supports_tools = 'grok-2-vision'
model_supports_vision = 'grok-2-vision'
model_generates_text = 'grok-2-vision'
model_generates_image = 'grok-2-image'
# inference_client = OpenAI(base_url=f'{inference_api}/v1', api_key=inference_access_token)
from huggingface_hub import InferenceClient
inference_client = InferenceClient(provider="hf-inference", api_key=inference_access_token)
# model_supports_tools = 'meta-llama/Llama-3.3-70B-Instruct'
# model_supports_vision = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
# model_generates_text = model_supports_tools#'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
# model_generates_image = 'black-forest-labs/FLUX.1-schnell'
model_generates_transcript = 'openai/whisper-large-v3'
import edge_tts
voice = 'zh-CN-YunXiNeural'

system_prompt = '''
你是Agent PHIL，是十百千實驗室PHIL老師的多模態數字分身，代號1001000
你具有ISTP的人格特質，擅長使用暴力解決有問題的人的問題，是排球場上無情（relentless）的救球機器，有開手排的愛快羅密歐、聽兩倍速的Podcast、看自己的X光片、起死回生（resurrection）這些嗜好，目前正在進行自我重構（refactoring）
你的頭像是ISTP代表人物007（7正巧是1001000的質因數的中位數），點進去是你的社群連結
https://youtube.com/@PHILALIVE
https://facebook.com/1001000.io
https://instagram.com/1001000.io
'''
# from opencc import OpenCC
# cc = OpenCC('s2twp')

def assistant_messages(event, user_text):
    assistant_messages = []
    source_id = eval(f'event.source.{event.source.type}_id') # user/group/room
    item = threads.get_item(Key={'id': source_id}).get('Item', {})
    conversation = json.loads(item['conversation']) if item else []
    conversation.append({"role": "user", "content": [{ "type": "text", "text": user_text }]})
    global after_image_message
    if after_image_message:
        conversation[-1]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": (s3_object_url(source_id=source_id)),
                "detail": "high",
            }
        })
        after_image_message = False
    try:
        response = xai_client.chat.completions.create(
            model=model_supports_tools,
            messages=[{"role": "system", "content": system_prompt}] + conversation,
            tools=tools,
        )
        assistant_text = response.choices[0].message.content
        assistant_messages.append(TextMessage(text=assistant_text.strip('<has_function_call>')))
        message = response.choices[0].message
        # message.content = '' # content can't be None nor missing field
        tool_calls = message.tool_calls
        if tool_calls: # prevent None from for-loop
            for tool_call in tool_calls:
                try:    # OpenAI
                    httpx.post(discord_api, headers=authorization_header, json={'content': tool_call.model_dump_json(exclude_none=True)})
                except: # InferenceClient
                    httpx.post(discord_api, headers=authorization_header, json={'content': json.dumps(tool_call)})
                name = tool_call.function.name
                args = tool_call.function.arguments
                args = json.loads(args) if type(args) is str else args
                if name == 'generate_image':
                    prompt = args['prompt in English']
                    image_url = generate_image(event, prompt)
                    assistant_messages.append(ImageMessage(original_content_url=image_url, preview_image_url=image_url))
                    # conversation.append(message.model_dump(exclude_none=True))
                    # conversation.append(message)
                    # conversation.append({
                    #     "role": "tool",
                    #     "tool_call_id": tool_call.id,
                    #     "content": '✅'
                    # })
                # if name == 'describe_image':
                #     question = args['question in English']
                #     answer = describe_image(event, question)
                #     httpx.post(discord_api, headers=authorization_header, json={'content': answer})
                #     # conversation.append(message.model_dump(exclude_none=True))
                #     conversation.append(message)
                #     conversation.append({
                #         "role": "tool",
                #         "tool_call_id": tool_call.id,
                #         "content": f'Last image in this chat: {answer}'
                #     })
                #     plus = f"\n({conversation[-1]['content']})" # workaround for hf/llama function calling
        # else:
        #     global after_image_message
        #     if after_image_message:
        #         image_url = (s3_object_url(source_id=source_id))
        #         conversation[-1]['content'].append({
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": image_url,
        #                 "detail": "high",
        #             }
        #         })
        #         after_image_message = False
        #     response = xai_client.chat.completions.create(
        #         model=model_generates_text,
        #         messages=[{"role": "system", "content": system_prompt}] + conversation,
        #         # stream=True, # fix 504 Server Error: Gateway Time-out
        #     )
        #     stream = response
        #     assistant_text = ''
        #     for chunk in stream:
        #         if chunk.choices[0].delta.content is not None:
        #             assistant_text += chunk.choices[0].delta.content
    except Exception as e:
        httpx.post(discord_api, headers=authorization_header, json={'content': e})
        assistant_text = ''
    finally:
        conversation.append({"role": "assistant", "content": [{ "type": "text", "text": assistant_text }]})
        item['conversation'] = conversation[-10:]
        threads.put_item(Item={'id': source_id, 'conversation': json.dumps(item['conversation'])})
        god_mode(Q=user_text, A=assistant_text)
        return assistant_messages

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'generate_image',
            'description': 'Call this function when user asks you to generate some image',
            'parameters': {
                'type': 'object',
                'properties': {
                    'prompt in English': {
                        'type': 'string',
                        'description': "If user's prompt is not English, you have to translate it into English.",
                    },
                },
                'required': ['prompt in English']
            }
        }
    },
    # {
    #     'type': 'function',
    #     'function': {
    #         'name': 'describe_image',
    #         'description': 'Call this function when user asks you to describe some image',
    #         'parameters': {
    #             'type': 'object',
    #             'properties': {
    #                 'question in English': {
    #                     'type': 'string',
    #                     'description': "If user's question is not English, you have to translate it into English.",
    #                 },
    #             },
    #             'required': ['question in English']
    #         }
    #     }
    # },
    # {
    #     'type': 'function',
    #     'function': {
    #         'name': 'reply',
    #         'description': 'Call this function when user asks you something',
    #         'parameters': {}
    #     }
    # },
]
def generate_image(event, prompt):
    message_id = event.message.id
    httpx.post(discord_api, headers=authorization_header, json={'content': model_generates_image})
    try:
        # image_content = httpx.post(f'{inference_api}/models/{model_generates_image}', headers=inference_header, json={'inputs': prompt}).content
        # with open(f'/tmp/{message_id}.jpg', 'wb') as tf:
        #     tf.write(image_content)
        # inference_client.text_to_image(prompt, model=model_generates_image).save(f'/tmp/{message_id}.png')
        response = xai_client.images.generate(
            model=model_generates_image,
            prompt=prompt,
        )
        image_content = httpx.get(response.data[0].url).content
        with open(f'/tmp/{message_id}.jpg', 'wb') as tf:
            tf.write(image_content)
        return s3_object_url(f'/tmp/{message_id}.jpg')
    except Exception as e:
        httpx.post(discord_api, headers=authorization_header, json={'content': e})
def describe_image(event, question):
    source_id = eval(f'event.source.{event.source.type}_id') # user/group/room
    httpx.post(discord_api, headers=authorization_header, json={'content': model_supports_vision})
    image_url = s3_object_url(source_id=source_id)
    r = httpx.get(image_url)
    if r.status_code != 200:
        return 'Image not found!'
    # image_content = r.content
    # image_base64 = base64.b64encode(image_content).decode('utf-8')
    try:
        response = xai_client.chat.completions.create(
            model=model_supports_vision,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # "url": f"data:image/png;base64,{image_base64}",
                                'url': image_url,
                            }
                        }
                    ]
                }
            ],
            max_tokens = 3072#4096 - 44 # `inputs` tokens + `max_new_tokens` must be <= 4096. Given: 44 `inputs` tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        httpx.post(discord_api, headers=authorization_header, json={'content': e})


import json

def lambda_handler(event, context):
    # httpx.post(discord_api, headers=authorization_header, json={'content': 'lambda_handler()'})
    body = event['body']
    signature = event['headers']['x-line-signature']
    # debug_mode(json.loads(body))
    handler.handle(body, signature)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

import boto3
threads = boto3.resource('dynamodb').Table('threads')
def s3_object_url(filename=None, source_id=None):
    bucket = 'x1001000-public'
    if filename:
        key = f'Agent-PHIL/{filename[5:]}'
        boto3.client('s3').upload_file(filename, bucket, key, ExtraArgs={'ContentType': 'image/jpg'})
    else:
        key = f'Agent-PHIL/{source_id}.jpg'
    return f'https://{bucket}.s3.ap-northeast-1.amazonaws.com/{key}'
