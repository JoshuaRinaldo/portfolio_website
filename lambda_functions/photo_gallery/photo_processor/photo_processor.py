"""
Photo Processing Lambda

Triggered when images are uploaded to s3://bucket/uploads/
Responsibilities:
1. Move original image to processed/full/ (preserving original quality and extension)
2. Create resized versions (thumbnail: 400px, medium: 1500px, large: 3000px as JPG)
3. Extract GPS coordinates from EXIF data
4. Call AWS Location Service for reverse geocoding (lat/lon → city, country)
5. Call AWS Rekognition for object/scene detection and dominant colors
6. Call AWS Bedrock (Claude) for AI-generated caption
7. Update gallery.json with photo metadata (images, location, caption, labels, colors)
8. Clean up uploads/ folder

Environment Variables:
- PHOTO_BUCKET: S3 bucket name
- BEDROCK_MODEL_ID: Bedrock model ID for captioning (optional)
- PLACE_INDEX_NAME: AWS Location Service place index name (optional)
"""

import boto3
import json
import os
import logging
import base64
from io import BytesIO
from PIL import Image, ImageOps
from urllib.parse import unquote_plus
from datetime import datetime
from typing import Dict

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
location_client = boto3.client('location', region_name='us-east-1')

# Configuration
PHOTO_BUCKET = os.environ['PHOTO_BUCKET']
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
PLACE_INDEX_NAME = os.environ.get('PLACE_INDEX_NAME', '')
GALLERY_JSON_KEY = 'metadata/gallery.json'

# Image sizes (max dimension)
SIZES = {
    'thumbnail': 400,
    'medium': 1500,
    'large': 3000,
}


def resize_image(image_bytes, max_dimension):
    """
    Resize image maintaining aspect ratio and correct orientation.

    Args:
        image_bytes: Original image bytes
        max_dimension: Maximum width or height

    Returns:
        tuple: (BytesIO of resized JPEG, PIL.Image object)
    """
    img = Image.open(BytesIO(image_bytes))

    # Fix orientation based on EXIF data (critical for photos from cameras/phones)
    img = ImageOps.exif_transpose(img)

    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background

    # Calculate new dimensions
    width, height = img.size
    if width > height:
        new_width = min(width, max_dimension)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, max_dimension)
        new_width = int(width * (new_height / height))

    # Resize
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save to bytes
    output = BytesIO()
    resized_img.save(output, format='JPEG', quality=95, optimize=True)
    output.seek(0)

    return output, resized_img


def detect_labels_rekognition(bucket: str, image_key: str) -> Dict:
    """
    Use AWS Rekognition to detect labels (objects, scenes) and dominant colors.

    Args:
        bucket: S3 bucket name
        image_key: S3 key for image

    Returns:
        Dict with labels, categories, and dominant colors
    """
    try:
        response = rekognition.detect_labels(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': image_key
                }
            },
            MaxLabels=20,
            MinConfidence=70.0,
            Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES'],
            Settings={
                'GeneralLabels': {
                    'LabelInclusionFilters': []
                },
                'ImageProperties': {
                    'MaxDominantColors': 10
                }
            }
        )

        # Extract labels and organize by category
        labels = []
        categories = set()

        for label in response.get('Labels', []):
            label_data = {
                'name': label['Name'],
                'confidence': round(label['Confidence'], 2),
                'categories': [cat['Name'] for cat in label.get('Categories', [])],
                'height': label['Height'],
                'left': label['Left'],
                'top': label['Top'],
                'width': label['Width'],
            }
            labels.append(label_data)
            categories.update(label_data['categories'])

        # Extract dominant colors from image properties
        image_properties = response.get('ImageProperties', {})
        dominant_colors = image_properties.get('DominantColors', [])

        # Format color data for easier frontend use
        colors = []
        for color in dominant_colors:
            colors.append({
                'hex': color.get('HexCode', ''),
                'rgb': {
                    'r': color.get('Red', 0),
                    'g': color.get('Green', 0),
                    'b': color.get('Blue', 0)
                },
                'css_color': color.get('CSSColor', ''),
                'simplified_color': color.get('SimplifiedColor', ''),
                'pixel_percentage': round(color.get('PixelPercent', 0), 2)
            })

        return {
            'labels': labels,
            'categories': list(categories),
            'label_count': len(labels),
            'dominant_colors': colors,
            'color_count': len(colors)
        }

    except Exception as e:
        logger.error(f"Rekognition error: {str(e)}", exc_info=True)
        return {
            'labels': [],
            'categories': [],
            'label_count': 0,
            'dominant_colors': [],
            'color_count': 0,
            'error': str(e)
        }


def extract_gps_from_exif(pil_image: Image.Image) -> Dict:
    """
    Extract GPS coordinates from image EXIF data.

    Args:
        pil_image: PIL Image object

    Returns:
        Dict with latitude, longitude, or empty dict if no GPS data
    """
    try:
        # img = pil_image.open()
        exif_data = pil_image.getexif()
        if not exif_data:
            return {}

        # EXIF GPS tags
        GPS_INFO = 34853

        gps_info = exif_data.get(GPS_INFO)
        if not gps_info:
            return {}

        # GPS sub-tags
        GPS_LATITUDE = 2
        GPS_LATITUDE_REF = 1
        GPS_LONGITUDE = 4
        GPS_LONGITUDE_REF = 3

        def convert_to_degrees(value):
            """Convert GPS coordinates to degrees"""
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0

        lat = gps_info.get(GPS_LATITUDE)
        lat_ref = gps_info.get(GPS_LATITUDE_REF)
        lon = gps_info.get(GPS_LONGITUDE)
        lon_ref = gps_info.get(GPS_LONGITUDE_REF)

        if lat and lon and lat_ref and lon_ref:
            latitude = convert_to_degrees(lat)
            if lat_ref == 'S':
                latitude = -latitude

            longitude = convert_to_degrees(lon)
            if lon_ref == 'W':
                longitude = -longitude

            return {
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6)
            }

        return {}

    except Exception as e:
        logger.warning(f"Could not extract GPS data: {str(e)}")
        return {}


def reverse_geocode_location(latitude: float, longitude: float) -> str:
    """
    Convert GPS coordinates to human-readable location using AWS Location Service.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Formatted location string like "Bangkok, Thailand" or "Sicily, Italy"
    """
    if not PLACE_INDEX_NAME:
        logger.info("No place index configured, skipping reverse geocoding")
        return ""

    try:
        response = location_client.search_place_index_for_position(
            IndexName=PLACE_INDEX_NAME,
            Position=[longitude, latitude],  # AWS Location uses [lon, lat] order
            MaxResults=1
        )

        if not response.get('Results'):
            return ""

        place = response['Results'][0]['Place']

        # Build location string: prefer "City, Country" or "Region, Country"
        city = place.get('Municipality') or place.get('Neighborhood')
        region = place.get('Region')
        country = place.get('Country')

        # Format location hierarchically
        if city and country:
            location_str = f"{city}, {country}"
        elif region and country:
            location_str = f"{region}, {country}"
        elif country:
            location_str = country
        else:
            location_str = ""

        logger.info(f"Reverse geocoded ({latitude}, {longitude}) → {location_str}")
        return location_str

    except Exception as e:
        logger.error(f"Reverse geocoding error: {str(e)}", exc_info=True)
        return ""


def generate_caption_bedrock(pil_image: Image.Image) -> str:
    """
    Generate image caption using AWS Bedrock.

    Args:
        pil_image: PIL Image object

    Returns:
        Generated caption string
    """
    try:

        # Convert PIL image to bytes and base64 encode
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64,
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please provide a short caption for this photo. The caption should be descriptive but it should not be embellished."
                        }
                    ]
                }
            ]
        }

        # Invoke Bedrock
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        caption = response_body['content'][0]['text'].strip()

        logger.info(f"Generated caption via Bedrock: {caption}")
        return caption

    except Exception as e:
        logger.error(f"Bedrock caption error: {str(e)}", exc_info=True)
        return ""


def load_gallery_json(bucket: str) -> Dict:
    """
    Load existing gallery.json from S3.

    Args:
        bucket: S3 bucket name

    Returns:
        Gallery data dict (or empty structure if not exists)
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=GALLERY_JSON_KEY)
        data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Loaded existing gallery.json with {len(data.get('photos', []))} photos")
        return data

    except s3.exceptions.NoSuchKey:
        logger.info("gallery.json not found, creating new one")
        return {
            'photos': [],
            'last_updated': None,
            'photo_count': 0
        }

    except Exception as e:
        logger.error(f"Error loading gallery.json: {str(e)}", exc_info=True)
        return {
            'photos': [],
            'last_updated': None,
            'photo_count': 0
        }


def save_gallery_json(bucket: str, gallery_data: Dict) -> None:
    """
    Save gallery.json to S3.

    Args:
        bucket: S3 bucket name
        gallery_data: Gallery data to save
    """
    try:
        # Update metadata
        gallery_data['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        gallery_data['photo_count'] = len(gallery_data.get('photos', []))

        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=GALLERY_JSON_KEY,
            Body=json.dumps(gallery_data, indent=2),
            ContentType='application/json',
            CacheControl='max-age=300'  # 5 minute cache
        )

        logger.info(f"Saved gallery.json with {gallery_data['photo_count']} photos")

    except Exception as e:
        logger.error(f"Error saving gallery.json: {str(e)}", exc_info=True)
        raise


def handler(event, context):
    """
    Lambda handler for complete photo processing pipeline.

    Triggered by S3 event when image is uploaded to uploads/ folder.
    """
    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])

            logger.info(f"Processing image: {key}")

            # Extract filename
            filename = os.path.basename(key)
            name, ext = os.path.splitext(filename)

            # Download original image
            response = s3.get_object(Bucket=bucket, Key=key)
            original_bytes = response['Body'].read()

            logger.info(f"Downloaded original image: {len(original_bytes)} bytes")

            # Move original to full/ folder (preserving original extension)
            full_key = f"processed/full/{name}{ext}"
            s3.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': key},
                Key=full_key,
                ContentType='image/jpeg',
                CacheControl='max-age=31536000',  # 1 year cache
                MetadataDirective='REPLACE'
            )
            logger.info(f"Moved original to full: {full_key}")

            # Process and upload resized versions
            processed_keys = {'full': full_key}
            large_pil_image = None  # Store large image for captioning

            for size_name, max_dim in SIZES.items():
                # Resize image
                resized_bytes, pil_img = resize_image(original_bytes, max_dim)

                # Upload to processed folder
                processed_key = f"processed/{size_name}/{name}.jpg"
                s3.put_object(
                    Bucket=bucket,
                    Key=processed_key,
                    Body=resized_bytes.getvalue(),
                    ContentType='image/jpeg',
                    CacheControl='max-age=31536000'  # 1 year cache
                )

                processed_keys[size_name] = processed_key
                logger.info(f"Uploaded {size_name}: {processed_key}")

                # Store large image for caption generation
                if size_name == 'large':
                    large_pil_image = pil_img

            # Detect labels with Rekognition
            logger.info("Calling Rekognition for label detection")
            rekognition_data = detect_labels_rekognition(bucket, processed_keys['large'])

            # Generate caption with Bedrock
            ai_caption = ""
            if large_pil_image:
                logger.info("Generating AI caption with Bedrock")
                ai_caption = generate_caption_bedrock(large_pil_image)

            # Extract GPS coordinates from EXIF and reverse geocode
            gps_data = {}
            location_string = ""
            if large_pil_image:
                logger.info("Extracting GPS data from EXIF")
                gps_data = extract_gps_from_exif(large_pil_image)

                if gps_data:
                    logger.info(f"Found GPS coordinates: {gps_data}")
                    location_string = reverse_geocode_location(
                        gps_data['latitude'],
                        gps_data['longitude']
                    )

            # Auto-generate title from top labels if no caption
            title = name.replace('_', ' ').replace('-', ' ').title()
            if not ai_caption and rekognition_data['labels']:
                top_labels = [l['name'] for l in rekognition_data['labels'][:3]]
                title = ' - '.join(top_labels)

            # Build photo metadata
            photo_metadata = {
                'id': name,
                'title': title,
                'caption': ai_caption,
                'date_uploaded': datetime.utcnow().isoformat() + 'Z',
                'images': {
                    'thumbnail': processed_keys['thumbnail'],
                    'medium': processed_keys['medium'],
                    'large': processed_keys['large'],
                    'full': processed_keys['full']
                },
                'rekognition': rekognition_data
            }

            # Add location data if available
            if gps_data:
                photo_metadata['location'] = {
                    'latitude': gps_data['latitude'],
                    'longitude': gps_data['longitude'],
                    'name': location_string
                }

            # Load gallery.json
            gallery_data = load_gallery_json(bucket)

            # Check if photo already exists (update) or add new
            existing_index = next(
                (i for i, p in enumerate(gallery_data['photos']) if p['id'] == name),
                None
            )

            if existing_index is not None:
                logger.info(f"Updating existing photo: {name}")
                gallery_data['photos'][existing_index] = photo_metadata
            else:
                logger.info(f"Adding new photo: {name}")
                gallery_data['photos'].append(photo_metadata)

            # Sort by date_uploaded (newest first)
            gallery_data['photos'].sort(
                key=lambda x: x.get('date_uploaded', ''),
                reverse=True
            )

            # Save gallery.json
            save_gallery_json(bucket, gallery_data)

            # Delete original from uploads/ folder
            s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted original: {key}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Photo processing complete',
                'photo_id': name,
                'title': title,
                'processed_keys': processed_keys,
                'labels_detected': rekognition_data['label_count'],
                'ai_caption_generated': bool(ai_caption)
            })
        }

    except Exception as e:
        logger.error(f"Error processing photo: {str(e)}", exc_info=True)
        raise