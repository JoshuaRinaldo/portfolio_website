#!/usr/bin/env python3
import os
import json

from aws_cdk import App, Environment

from cdk_stack.static_site_stack import StaticSite
# from cdk_stack.streamlit_site_stack import StreamlitSite  # Old stack - kept for reference

with open("cdk.json", "r") as file:
    context = json.load(file)["context"]

account = context["account"]
region = context["region"]

app = App()

# New static site stack - cost-effective S3 + CloudFront + API Gateway
StaticSite(
    app,
    "StaticSite",
    env=Environment(account=account, region=region)
)

# Old Streamlit stack - commented out, remove after successful migration
# StreamlitSite(
#     app,
#     "StreamlitSite",
#     env=Environment(account=account, region=region)
# )

app.synth()
