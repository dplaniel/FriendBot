import discord
from discord.ext import commands
import pandas as pd
import datetime
import pickle
from tqdm.asyncio import tqdm

from friendbot_config import FRIENDBOT_TOKEN, PRIVILEGED_USERS


bot = commands.Bot(command_prefix="$")


@bot.command()
async def collect(ctx):
    if ctx.author.id in PRIVILEGED_USERS:
        with open(ctx.channel.name + ".pkl", "wb") as outfile:
            print("Retrieving...")
            # i = 0
            msg_ids = []
            auth_ids = []
            auth_nms = []
            auth_bots = []
            contents = []
            cln_contents = []
            dts = []
            async for message in tqdm(ctx.history(limit=None, oldest_first=True)):
                msg_ids.append(message.id)
                auth_ids.append(message.author.id)
                auth_nms.append(message.author.name)
                auth_bots.append(message.author.bot)
                contents.append(message.content)
                cln_contents.append(message.clean_content)
                dts.append(message.created_at)
            print("Retrieved!")
            # print(message.author.name, message.content)
            columns_list = [
                "message_id",
                "author_id",
                "author_name",
                "bot",
                "content",
                "clean_content",
                "created_at",
            ]
            df = pd.DataFrame(
                {
                    "message_id": msg_ids,
                    "author_id": auth_ids,
                    "author_name": auth_nms,
                    "bot": auth_bots,
                    "content": contents,
                    "clean_content": cln_contents,
                    "created_at": dts,
                }
            )
            print(df.head(5))
            pickle.dump(df, outfile)


bot.run(FRIENDBOT_TOKEN)
