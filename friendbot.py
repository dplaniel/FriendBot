import discord
from discord.ext import commands
import asyncio
from random import randint
import os
import unicodedata

from friendbot_config import FRIENDBOT_TOKEN, PRIVILEGED_USERS, STABLEDIFFUSION_LOCATION

# Create Bot object from Discord Commands API
bot = commands.Bot(command_prefix="!")


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
        unicodedata.normalize("NFKD", prompt).encode("ascii", "replace").decode()
    )

    if (seed is None) or (seed not in range(1, 64000)):
        seed = randint(1, 64000)
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
        print(f"Stable diffusion returned with non-zero exit code {proc.returncode}")
        with open("err_logs.txt", "w+") as errfile:
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


@bot.command()
async def lonely(ctx):
    # First check if there is enough GPU Memory and power available:
    is_gpu_available = await check_gpu_availability(3200)
    if not is_gpu_available:
        await ctx.send(
            f"I need a lot of free GPU memory to do that, and it looks like there isn't enough to go around.  Try again later."
        )
    else:
        await ctx.send("Sorry, `!lonely` is not implemented yet.")


@bot.command()
async def artistic(ctx, *, prompt):
    if (
        not ctx.channel.name == "artism-spectrum"
    ):  # temporary hack while I think of a better way to specify by channel id
        return
    # First check if there is enough GPU Memory and power available:
    is_gpu_available = await check_gpu_availability()
    if not is_gpu_available:
        await ctx.reply(
            f"I need a lot of free GPU memory to do that, and it looks like there isn't enough to go around.  Try again later."
        )
        return
    else:
        await ctx.reply("Okay, give me a minute to try to draw that!")
    if prompt.startswith('"'):
        prompt = prompt[1:]
    if prompt.endswith('"'):
        prompt = prompt[:-1]
    imgpath = await run_stable_diffusion(prompt, label=True)  # label=False)
    if imgpath is not None:
        imgfile = discord.File(imgpath)
        await ctx.reply("How's this?", file=imgfile)
    else:
        await ctx.reply("I think something went wrong, sorry.")


bot.run(FRIENDBOT_TOKEN)
