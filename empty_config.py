import os

# Empty prototype for friendbot_config.py
# Please populate this with the necessary values for your bot/server and local installation details
# before running; save out as friendbot_config.py


FRIENDBOT_TOKEN = ""  # Put your bot token here
PRIVILEGED_USERS = []  # List the unique User IDs of Friendbot administrators here
STABLEDIFFUSION_LOCATION = os.path.abspath(
    os.path.expandvars("/path/to/stable-diffusion/install/location")
)  # if using optional StableDiffusion image generation, include path to your SD install location here
# see: https://github.com/CompVis/stable-diffusion
