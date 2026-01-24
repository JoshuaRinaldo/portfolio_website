from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import Rekognition, Bedrock
from diagrams.aws.network import CloudFront, Route53
from diagrams.aws.general import User
from diagrams.onprem.client import Client

# Custom attributes for styling
graph_attr = {
    "fontsize": "18",
    "bgcolor": "white",
    "pad": "1",
}

with Diagram(
    "Photo Gallery Architecture",
    filename="photo_gallery_architecture",
    outformat="png",
    show=False,
    graph_attr=graph_attr
):

    photographer = User("")
    website_visitor = Client("Website Visitor")

    with Cluster("AWS Cloud"):
        photo_bucket = S3("Photo Bucket")
        photo_processor = Lambda("Photo Processor")

        with Cluster("AI Services"):
            bedrock = Bedrock("Bedrock")
            rekognition = Rekognition("Rekognition")

        with Cluster("Website"):
            cloudfront = CloudFront("CloudFront")
            website_bucket = S3("Website Bucket")
            route53 = Route53("Route 53")

    # Upload & Processing Flow
    photographer >> Edge(label="1. Upload photo") >> photo_bucket
    photo_bucket >> Edge(label="2. Event Trigger") >> photo_processor

    photo_processor >> Edge(label="3. Generate caption") >> bedrock
    photo_processor >> Edge(label="4. Detect labels/colors") >> rekognition

    photo_processor >> Edge(label="5. Save processed\nphotos") >> photo_bucket

    # Website Delivery Flow
    website_visitor >> route53
    route53 >> cloudfront
    cloudfront >> website_bucket
    cloudfront >> Edge(label="Fetch photos & metadata") >> photo_bucket