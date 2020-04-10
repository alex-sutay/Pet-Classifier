"""
author: Alex Sutay
file: betbot.py
"""

import discord
import predictImage
import pic_to_vect
from PIL import Image
from config import TOKEN, THETA_FILE, CHANNEL, KEY

client = discord.Client()


@client.event
async def on_message(message):
    for attachment in message.attachments:
        await message.channel.send("Hmm, let me think for a moment...")
        await attachment.save("image")
        im = Image.open("image")
        functs = {'r': pic_to_vect.to_nums_r, 'g': pic_to_vect.to_nums_g,
                  'b': pic_to_vect.to_nums_b,  'l': pic_to_vect.to_nums_l}
        im_array = functs[CHANNEL](im, pic_to_vect.SIZE, pic_to_vect.SIZE)
        thetas = predictImage.thetas_from_mat(THETA_FILE)
        pred = predictImage.predict(thetas, im_array)
        await message.channel.send("Is it possibly " + KEY[pred] + "?")


@client.event
async def on_ready():
    print("Logged in as")
    print(client.user.name)
    print(client.user.id)


client.run(TOKEN)
