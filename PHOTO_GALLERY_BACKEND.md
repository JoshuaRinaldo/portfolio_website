# Photo Gallery Backend - Implementation Guide

## Overview

Automated photo captioning and gallery management system using AWS services.

## Architecture

```
User uploads → S3 (uploads/)
    ↓ (S3 Event Trigger)
ProcessImageLambda
    ├─ Creates 4 sizes: thumbnail (200px), medium (800px), large (2000px), full (original)
    ├─ Uploads to processed/
    └─ Invokes GenerateCaptionLambda
        ↓
GenerateCaptionLambda
    ├─ AWS Rekognition → Object/scene detection
    ├─ SageMaker Endpoint → AI caption generation (optional)
    ├─ Updates gallery.json
    └─ Cleanup: deletes from uploads/
        ↓
Frontend fetches gallery.json + displays photos
```

## S3 Bucket Structure

```
{environment}-photo-gallery/
├── uploads/                    # Temporary staging (auto-deleted after processing)
├── processed/
│   ├── thumbnail/             # 200px - Grid view
│   ├── medium/                # 800px - Preview
│   ├── large/                 # 2000px - Full screen viewing (website)
│   └── full/                  # Original dimensions - Download only
└── metadata/
    └── gallery.json           # Photo metadata with captions & labels
```

## gallery.json Schema

```json
{
  "photos": [
    {
      "id": "photo1",
      "title": "Mountain Lake Sunset",
      "caption": "AI-generated caption here",
      "date_uploaded": "2025-12-31T12:00:00Z",
      "images": {
        "thumbnail": "processed/thumbnail/photo1.jpg",
        "medium": "processed/medium/photo1.jpg",
        "large": "processed/large/photo1.jpg",
        "full": "processed/full/photo1.jpg"
      },
      "rekognition": {
        "labels": [
          {
            "name": "Mountain",
            "confidence": 98.5,
            "categories": ["Nature", "Outdoors"]
          }
        ],
        "categories": ["Nature", "Outdoors"],
        "label_count": 15
      }
    }
  ],
  "last_updated": "2025-12-31T12:00:00Z",
  "photo_count": 1
}
```

## Deployment Steps

### 1. Add Photo Gallery Stack to app.py

```python
from cdk_stack.photo_gallery_stack import PhotoGalleryStack

# In your app.py, add:
photo_gallery_stack = PhotoGalleryStack(
    app,
    f"{environment}-photo-gallery",
    caption_endpoint_name="image-caption-generator",  # Optional
    env=cdk.Environment(account=account, region=region)
)
```

### 2. (Optional) Add Image Captioning SageMaker Endpoint

Add to `cdk.json` → `sagemaker_endpoints`:

```json
{
  "endpoint_type": "huggingface",
  "environment_variable_name": "IMAGE_CAPTION_MODEL",
  "endpoint_name": "image-caption-generator",
  "model_task": "image-to-text",
  "serverless_config": {
    "memory_size_in_mb": 6144,
    "max_concurrency": 1
  },
  "model_name": "Salesforce/blip-image-captioning-base"
}
```

**Recommended Models:**
- `Salesforce/blip-image-captioning-base` - Good quality, efficient
- `Salesforce/blip-image-captioning-large` - Higher quality, slower
- `nlpconnect/vit-gpt2-image-captioning` - Alternative option

### 3. Deploy

```bash
cd /Users/jrinaldo/code/aws_cdk_tutorial/portfolio_website
cdk deploy
```

### 4. Upload Photos

**Option A: AWS Console**
1. Go to S3 → `{environment}-photo-gallery`
2. Navigate to `uploads/` folder
3. Upload photos (JPG, PNG supported)
4. Processing triggers automatically

**Option B: AWS CLI**
```bash
aws s3 cp /path/to/photo.jpg s3://{environment}-photo-gallery/uploads/photo.jpg
```

**Option C: Python Script (Bulk Upload)**
```python
import boto3

s3 = boto3.client('s3')
bucket = 'prod-photo-gallery'

photos = [
    '/path/to/photo1.jpg',
    '/path/to/photo2.jpg',
]

for photo_path in photos:
    filename = os.path.basename(photo_path)
    s3.upload_file(
        photo_path,
        bucket,
        f'uploads/{filename}'
    )
    print(f'Uploaded: {filename}')
```

## Frontend Integration

### Fetch Gallery Data

```javascript
// Fetch gallery.json
const GALLERY_URL = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com/metadata/gallery.json';

async function loadGallery() {
    const response = await fetch(GALLERY_URL);
    const data = await response.json();
    return data.photos;
}

// Display in gallery
const photos = await loadGallery();
photos.forEach(photo => {
    // Use photo.images.thumbnail for grid
    // Use photo.images.large for lightbox
    // Use photo.images.full for download link
});
```

### Example Gallery Item

```html
<div class="gallery-item">
    <img src="${photo.images.thumbnail}" alt="${photo.title}">
    <h3>${photo.title}</h3>
    <p>${photo.caption}</p>

    <!-- Object detection toggle -->
    <button onclick="showLabels('${photo.id}')">
        View Object Detection
    </button>

    <!-- Download full resolution -->
    <a href="${photo.images.full}" download>
        Download Full Resolution
    </a>
</div>
```

## Cost Estimates

**Storage (S3):**
- 100 photos × 4 sizes × ~500KB avg = ~200MB
- Cost: ~$0.005/month

**Processing (Lambda):**
- 100 photos processed = 200 Lambda invocations
- Cost: ~$0.01 (one-time)

**Rekognition:**
- 100 images × $0.001 = $0.10 (one-time)

**SageMaker Serverless (if enabled):**
- Minimal with serverless, only pay per invocation
- ~$0.10-0.20 for 100 captions

**Total Monthly: <$1** (after initial processing)

## Monitoring

### Check Processing Status

```bash
# View Lambda logs
aws logs tail /aws/lambda/prod-process-image --follow

aws logs tail /aws/lambda/prod-generate-caption --follow
```

### Verify gallery.json

```bash
aws s3 cp s3://prod-photo-gallery/metadata/gallery.json - | jq .
```

## Troubleshooting

### Photo not processing?
1. Check S3 event notification is configured
2. Check Lambda execution role has Rekognition permissions
3. View CloudWatch logs for errors

### Rekognition not detecting labels?
- Ensure image is in supported format (JPG, PNG)
- Check image quality and size
- Review MinConfidence threshold (currently 70%)

### SageMaker caption not generating?
- Verify endpoint is deployed and in service
- Check endpoint name environment variable
- Review memory allocation (may need to increase)

## Next Steps

1. ✅ Backend infrastructure complete
2. 🔄 Add SageMaker caption endpoint (optional)
3. 📸 Upload test photos
4. 🎨 Build frontend gallery page
5. 🚀 Deploy and test end-to-end

## Important: Stack Deletion & Redeployment

The photo bucket uses `RemovalPolicy.RETAIN` to protect your photos. Here's what happens:

### Normal Deletion
```bash
cdk destroy
```
**Result**: Stack deleted, **but bucket and all photos remain in AWS** ✅

### Redeploying After Deletion

If you delete the stack and want to redeploy:

1. **Bucket still exists** with all your photos
2. **Update cdk.json**:
   ```json
   "photo_bucket_exists": true
   ```
3. **Redeploy**:
   ```bash
   cdk deploy
   ```
4. Stack reconnects to existing bucket, preserves all data ✅

### Why This Pattern?

- **Prevents accidental data loss** - explicit flag required to reattach
- **Clear intent** - anyone reading code knows bucket pre-exists
- **Fail-safe** - CDK will error if you try to create duplicate bucket
- **Standard AWS CDK practice** for stateful resources

## Security Notes

- **Bucket Policy**: Only `processed/` and `metadata/` are publicly readable
- **Upload Security**: Consider adding Lambda authorizer if you want web-based uploads
- **Data Retention**: Bucket has RETAIN policy - data persists even if stack deleted
- **Redeployment Safety**: Manual flag prevents accidental bucket recreation
