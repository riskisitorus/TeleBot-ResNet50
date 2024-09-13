from pyrogram import filters , Client
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from dotenv import load_dotenv
import os
# import PIL.Image
import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50

load_dotenv()

API_HASH  = os.getenv('API_HASH')
API_ID  = os.getenv('API_ID')
BOT_TOKEN  = os.getenv('BOT_TOKEN')

app= Client('VisionScriptBot',
            api_hash=API_HASH,api_id=int(API_ID),
            bot_token=BOT_TOKEN)

# your_model_path = 'model_grid_search.h5'  # Nama file model Anda
# model = load_model(your_model_path)

# Load model ResNet50
model = ResNet50(weights='imagenet')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    for class_id, class_name, class_prob in decoded_preds:
        result = f'{class_name}: {class_prob:.2f}'
        return result

@app.on_message(filters.command('start') & filters.private)
async def start(_,message:Message):
    welcome_message = (
        f" Halo @{message.chat.username}!\n\n"
        "Ada yang bisa saya bantu?\n\n"
        "Kirimkan saja gambar yang ingin anda kenali, saya akan coba membantu anda 🤖"
    )
    await message.reply(welcome_message,quote=True)

@app.on_message(filters.photo & filters.private)
async def vision(bot,message:Message):
    try:
        txt = await message.reply(f'Loading...')
        await txt.edit('Downloading Photo ....')
        file_path = await message.download()
        # img = PIL.Image.open(file_path)
        await txt.edit('Gambar sedang diproses...')
        response = classify_image(file_path)
        await txt.delete()
        # os.remove(file_path)
        if response:
            await message.reply(response)
        else:
            await message.reply('Gambar tidak berhasil dikenali.')
    except Exception as e:
        await message.reply('Gambar gagal di proses!')
        raise e


app.run(print('Bot Started...'))