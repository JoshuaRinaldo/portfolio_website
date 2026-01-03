"""
Photo Gallery Stack

Creates infrastructure for automated photo captioning and gallery management:
- S3 bucket for photo storage (uploads/, processed/, metadata/)
- Lambda for image processing (resize, thumbnails)
- Lambda for caption generation (Rekognition + AWS Bedrock)
- S3 event triggers for automated processing

Note: Photo bucket uses RETAIN removal policy to preserve photos even if stack is deleted.
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_iam as iam,
    aws_location as location,
)
from .lambda_ import LambdaFunctionFromDockerImage
from constructs import Construct


class PhotoGalleryStack(Stack):
    """
    Photo Gallery Stack for automated image captioning and processing.

    Architecture:
    1. Upload image → s3://bucket/uploads/
    2. S3 trigger → PhotoProcessorLambda
       - Creates thumbnails (400px, 1500px, 3000px)
       - Calls AWS Rekognition for object/scene detection and dominant colors
       - Calls AWS Bedrock for AI-generated caption
       - Updates gallery.json metadata file
       - Moves processed images to processed/
       - Deletes original from uploads/
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        bedrock_model_id: str = "",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        environment = self.node.try_get_context("environment")
        region = self.node.try_get_context("region")
        photo_bucket_exists = self.node.try_get_context("photo_bucket_exists") or False

        bucket_name = f"{environment}-photo-gallery"

        # Import existing bucket or create new one based on context flag
        # If stack was deleted but bucket retained, set photo_bucket_exists=true in cdk.json
        if photo_bucket_exists:
            # Import existing bucket (read-only from CDK perspective)
            photo_bucket = s3.Bucket.from_bucket_name(
                self,
                f"{environment}-photo-bucket",
                bucket_name=bucket_name
            )
        else:
            # Create new S3 bucket for photos with RETAIN policy
            photo_bucket = s3.Bucket(
                self,
                f"{environment}-photo-bucket",
                bucket_name=bucket_name,
            # Enable CORS for frontend access
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                    allowed_origins=["*"],  # Restrict to your domain in production
                    allowed_headers=["*"],
                    max_age=3600
                )
            ],
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=False,
                block_public_policy=False,
                ignore_public_acls=False,
                restrict_public_buckets=False
            ),
                # RETAIN policy: bucket and data persist even if stack is deleted
                removal_policy=RemovalPolicy.RETAIN,
                auto_delete_objects=False,  # Keep objects when stack deleted
            )

            # Bucket policy to allow public read for processed/ folder only
            # Only add policy if creating new bucket (imported buckets already have policies)
            photo_bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    sid="PublicReadProcessedPhotos",
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AnyPrincipal()],
                    actions=["s3:GetObject"],
                    resources=[
                        photo_bucket.arn_for_objects("processed/*"),
                        photo_bucket.arn_for_objects("metadata/*")
                    ]
                )
            )

        # AWS Location Service Place Index for reverse geocoding (lat/lon → city, country)
        place_index = location.CfnPlaceIndex(
            self,
            f"{construct_id}-place-index",
            index_name=f"{environment}-photo-locations",
            data_source="Esri",  # Using Esri as the data provider (free tier available)
            description="Place index for reverse geocoding photo locations"
        )

        # Lambda function for complete photo processing pipeline
        # Handles: resize, thumbnails, Rekognition, Bedrock captioning, location extraction, gallery.json updates
        photo_processor_policy_statements = [
            {
                "resources": [photo_bucket.bucket_arn, f"{photo_bucket.bucket_arn}/*"],
                "actions": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ]
            },
            {
                "resources": ["*"],
                "actions": [
                    "rekognition:DetectLabels",
                    "rekognition:DetectText",
                ]
            },
            {
                "resources": ["*"],
                "actions": ["bedrock:InvokeModel"]
            },
            {
                "resources": [place_index.attr_arn],
                "actions": [
                    "geo:SearchPlaceIndexForPosition"
                ]
            }
        ]

        # Prepare environment variables
        lambda_environment = {
            "PHOTO_BUCKET": photo_bucket.bucket_name,
            "PLACE_INDEX_NAME": place_index.index_name,
        }

        # Add Bedrock model ID if provided
        if bedrock_model_id:
            lambda_environment["BEDROCK_MODEL_ID"] = bedrock_model_id

        photo_processor_lambda = LambdaFunctionFromDockerImage(
            scope=self,
            construct_id=f"{construct_id}-photo-processor",
            lambda_folder="photo_gallery/photo_processor",
            platform="arm64",
            timeout=3,  # minutes
            memory_size=2048,
            policy_statements=photo_processor_policy_statements,
            environment=lambda_environment
        )

        # S3 event notification: trigger photo_processor_lambda on uploads/
        # Only add if creating new bucket (imported buckets are read-only from CDK)
        if not photo_bucket_exists:
            photo_bucket.add_event_notification(
                s3.EventType.OBJECT_CREATED,
                s3n.LambdaDestination(photo_processor_lambda.lambda_function),
                s3.NotificationKeyFilter(prefix="uploads/")
            )

        CfnOutput(
            self,
            "PhotoBucketName",
            value=photo_bucket.bucket_name,
            description="Photo storage bucket name"
        )

        CfnOutput(
            self,
            "PhotoBucketUrl",
            value=f"https://{photo_bucket.bucket_name}.s3.{region}.amazonaws.com",
            description="Photo bucket URL"
        )

        CfnOutput(
            self,
            "GalleryDataUrl",
            value=f"https://{photo_bucket.bucket_name}.s3.{region}.amazonaws.com/metadata/gallery.json",
            description="Gallery metadata JSON URL"
        )

        # Store for use in other stacks
        self.photo_bucket = photo_bucket
        self.photo_bucket_name = photo_bucket.bucket_name
