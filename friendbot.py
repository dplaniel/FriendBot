import discord
from discord.ext import tasks, commands
import asyncio
from random import randint, choice
import os
import unicodedata
from collections import defaultdict

from friendbot_config import FRIENDBOT_TOKEN, PRIVILEGED_USERS, STABLEDIFFUSION_LOCATION
from dialog import channel_slowdown, channel_queue_full, user_at_cap, snarky_channel_slowdowns, snarky_usercaps


async def run_cmd(cmd):
    """
    AsyncIO wrapper for trusted, (implemented here, not sourced from user commands) shell commands
    """
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()

    if stderr:
        raise OSError(stderr.decode())
    assert (
        proc.returncode == 0
    ), f"{cmd} returned with non-zero exit code {proc.returncode}"

    return stdout.decode()


async def check_gpu_availability(req_gpu_mem_mb=10400):
    """
    Poll nvidia-smi to make sure the desired amount of GPU Memory is available
    """
    memstr = await run_cmd("nvidia-smi --query-gpu=memory.free --format=csv,noheader")
    try:
        memmb = int(memstr.split(" MiB")[0])
    except ValueError as e:
        print(f"Bad memory string return value: {memstr}")
        memmb = -1

    return memmb > req_gpu_mem_mb


async def run_stable_diffusion(
    prompt,
    seed=None,
    n_samples=1,
    n_iter=1,
    scale=7.2,
    ddim_steps=64,
    plms=True,
    label=False,
    outdir="/data/big/sdiff_images/friendbot",
    skip_grid=True,
):
    """
    Run stable-diffusion in subprocess (using sd's default "ldm" conda environment)
    and, if successful, return file path to image output

    See stable-diffusion/scripts/txt2img.py for implementation of the interface we
    are calling here

    Args:
        prompt : str, input prompt for txt2img.py
        seed : int, random seed for txt2img.py
        n_samples : int, Batch size for txt2img.py
        n_iter : int, number of batched repetitions to run in txt2img.py
        scale : float, CFG scale factor for sampling method in txt2img.py
        ddim_steps : int, number of steps in txt2img.py
        plms : bool, flag for whether or not to use plms sampling in txt2img.py
        label : bool, flag for img metadata bottom banner on txt2img.py output
            (requires custom fork of stable-diffusion...)
        outdir : str, path to output location
        skip_grid : bool, whether to output grid of batched outputs in txt2img.py
    Returns:
        str : Path to output .png location
    """
    safe_prompt = (
        unicodedata.normalize("NFKD", prompt).encode(
            "ascii", "replace").decode()
    )

    if (seed is None) or (not isinstance(seed, int)):
        seed = randint(1, 2**63-1)
    seed = max(0, min(seed, 2**63-1))
    ldm_python_path = os.path.abspath(
        os.path.expandvars("$HOME/anaconda3/envs/ldm/bin/python")
    )
    sd_cmd = [
        ldm_python_path,
        "-W",
        "ignore",
        f"{STABLEDIFFUSION_LOCATION}/scripts/txt2img.py",
        "--prompt",
        safe_prompt,
        "--n_samples",
        str(int(n_samples)),
        "--seed",
        str(int(seed)),
        "--n_iter",
        str(int(n_iter)),
        "--scale",
        str(scale),
        "--ddim_steps",
        str(int(ddim_steps)),
        "--outdir",
        outdir,
    ]
    if plms:
        sd_cmd.append("--plms")
    if label:
        sd_cmd.append("--label")
    if skip_grid:
        sd_cmd.append("--skip_grid")

    cwd = os.path.abspath(os.curdir)
    os.chdir(STABLEDIFFUSION_LOCATION)
    os.environ["PYTHONWARNINGS"] = "ignore"

    existing_images = os.listdir(os.path.join(outdir, "samples"))
    if len(existing_images) == 0:
        next_idx = "00000"
    else:
        last_img = sorted(existing_images)[-1]
        next_idx = format(int(last_img.split(".png")[0]) + 1, "05d")

    proc = await asyncio.create_subprocess_exec(
        *sd_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    # if stderr: # Python shits all of its warnings out onto stderr, so...
    #    raise OSError(stderr.decode())
    if proc.returncode != 0:
        print(
            f"Stable diffusion returned with non-zero exit code {proc.returncode}")
        with open("err_logs.txt", "a") as errfile:
            errfile.write(f"{next_idx}: {prompt}")
            errfile.write(f"\t {proc.returncode}")
            errfile.write(f"\t {stderr.decode()}")
            errfile.write(("\n"))

    imgfile = os.path.join(outdir, "samples", f"{next_idx}.png")

    os.chdir(cwd)
    os.environ["PYTHONWARNINGS"] = "default"

    if os.path.exists(imgfile):
        return imgfile
    else:
        print("Something went wrong. :(")  # This is bad error handling lol
        return None


class TextGenerationCog(commands.Cog):
    # TODO: Language model text generation -- the real puzzle here is GPU
    # resource balancing between this task and the Image task...
    def __init__(self, bot):
        pass

    @commands.command()
    async def lonely(ctx):
        # First check if there is enough GPU Memory and power available:
        is_gpu_available = await check_gpu_availability(3200)
        if not is_gpu_available:
            await ctx.send(
                f"I need a lot of free GPU memory to do that, and it looks like there isn't enough to go around.  Try again later."
            )
        else:
            await ctx.send("Sorry, `!lonely` is not implemented yet.")


class ImageGenerationCog(commands.Cog):
    """
    Cog to add Stable Diffusion txt2img.py Image Generation
    to a discord bot.

    Holds queue of user prompts in a prompt_queue.
    Bots with this cog will asynchronously spawn up to
    one python subprocess to run txt2img.py and process
    prompts.

    Users may submit prompts through the command interface
    via
        $artistic <user_prompt_here>

    Bots with this cog will reply directly to the message
    creating the prompt when the image is ready
    """

    def __init__(self, bot, max_queue_size=99, delete_images=False, snarky=True):
        """
        ImageGenerationCog Constructor

        Args:
            bot: commands.Bot Object
                Bot instance this Cog will be injected into
            max_queue_size: int, optional
                Max number of prompts in queue, defaults to 99
            delete_images : bool, optional
                Set this flag to True if you want to delete
                Stable Diffusion output images after they are
                uploaded to preserve disk space
            snarky: bool, optional
                Gives users snarky slowdown messages when they
                hit their prompt cap or the queue fills
        """
        self.bot = bot
        self.prompt_queue = asyncio.Queue(max_queue_size)
        self.retrieve_prompt_from_queue.start()
        self.delete_images = delete_images
        self.user_prompt_counts = defaultdict(lambda: 0)
        self.snarky = snarky
        self.slow_cap = .50

    @tasks.loop(seconds=0.25)
    async def retrieve_prompt_from_queue(self):
        """
        Retrieve prompts to run from queue, looped
        """
        (ctx, prompt) = await self.prompt_queue.get()  # Should wait until job is available
        self.user_prompt_counts[ctx.author] -= 1
        # First check if there is enough GPU Memory available:
        is_gpu_available = await check_gpu_availability()
        if not is_gpu_available:
            await ctx.reply(
                f"I need a lot of free GPU memory to do that, and it looks like there isn't enough to go around right now.  Try again later."
            )
            await asyncio.sleep(0.1)
            return
        imgpath = await run_stable_diffusion(prompt, label=True)
        if imgpath is not None:
            imgfile = discord.File(imgpath)
            await ctx.reply("How's this?", file=imgfile)
            if self.delete_images:
                os.remove(imgpath)
        else:
            await ctx.reply("I think something went wrong, sorry.")

    @commands.command()
    async def artistic(self, ctx, *, prompt):
        # Check we are in allowed channel
        if (
            not ctx.channel.name == "artism-spectrum"
        ):  # temporary hack while I think of a better way to specify by channel id
            return
        # Check there is space in the queue
        if self.prompt_queue.full():
            await ctx.reply(channel_queue_full)
            return
        # Clean prompt -- Remove trailing whitespace and quotes
        prompt = prompt.strip(" \t\n\'\"")
        # Check against per-user prompt caps
        usr_ct = self.user_prompt_counts[ctx.author]
        if usr_ct > 5:
            if self.snarky:
                reply = choice(snarky_usercaps).format(
                    member=ctx.author, plus_delimited_prompt="+".join(prompt.split()))
            else:
                reply = user_at_cap.format(member=ctx.author, usercap=5)
            await ctx.reply(reply)
            return
        await ctx.reply(f"Okay, I have {self.prompt_queue.qsize()+1} prompt(s) in the queue. {usr_ct+1} from {ctx.author}.")
        self.user_prompt_counts[ctx.author] += 1
        # Slowdown warning if approaching prompt limit
        if self.prompt_queue.qsize() >= int(self.slow_cap * self.prompt_queue.maxsize):
            if self.snarky:
                reply = choice(snarky_channel_slowdowns).format(n_slow_cap=int(
                    self.slow_cap*self.prompt_queue.maxsize), cmd_prefix=self.bot.command_prefix)
            else:
                reply = channel_slowdown.format(
                    slow_cap=int(100*self.slow_cap))
            await ctx.channel.send(reply)
        try:
            self.prompt_queue.put_nowait((ctx, prompt))
        except QueueFull:
            ctx.reply(channel_queue_full)


if __name__ == "__main__":

    # Create bot and add our
    friendbot = commands.Bot(command_prefix="!")
    friendbot.add_cog(ImageGenerationCog(
        friendbot, snarky=True, max_queue_size=16))

    # bot.run() is blocking and must be run last
    friendbot.run(FRIENDBOT_TOKEN)
