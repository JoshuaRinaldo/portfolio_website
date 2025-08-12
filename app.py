#!/usr/bin/env python3
import os
import json

from aws_cdk import App, Environment

from cdk_stack.streamlit_site_stack import StreamlitSite

with open("cdk.json", "r") as file:
    context = json.load(file)["context"]

account = context["account"]
region = context["region"]

app = App()
StreamlitSite(
    app,
    "StreamlitSite",
    env=Environment(account=account, region=region)
    )

app.synth()
