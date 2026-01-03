#!/usr/bin/env python3
import os
import json

from aws_cdk import App, Environment

from cdk_stack.static_site_stack import StaticSite
from cdk_stack.photo_gallery_stack import PhotoGalleryStack

with open("cdk.json", "r") as file:
    context = json.load(file)["context"]

account = context["account"]
region = context["region"]
env = context["environment"]


app = App()

# New static site stack - cost-effective S3 + CloudFront + API Gateway
StaticSite(
    app,
    "StaticSite",
    env=Environment(account=account, region=region)
)

photo_gallery_stack = PhotoGalleryStack(
    app,
    f"photo-gallery-{env}",
    env=Environment(account=account, region=region)
)


app.synth()
