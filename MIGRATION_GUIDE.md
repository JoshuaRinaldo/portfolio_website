# Portfolio Website Migration Guide
## From Streamlit/ECS to Static HTML/JS

## Overview

This migration transforms your portfolio website from an expensive always-on ECS/ALB infrastructure to a cost-effective serverless architecture using S3, CloudFront, and API Gateway.

### Cost Comparison

**Before (Streamlit/ECS):**
- Application Load Balancer: ~$16-20/month
- NAT Gateway: ~$32/month + data transfer
- ECS Fargate: ~$12-15/month
- **Total: ~$60-70/month minimum**

**After (Static Site):**
- S3 static hosting: ~$0.50-2/month
- CloudFront CDN: ~$1-3/month
- API Gateway: ~$3.50/month for 1M requests
- Lambda + SageMaker: Same as before (already serverless)
- **Total: ~$5-10/month**

**Savings: ~$50-60/month (~85% cost reduction)**

---

## What Changed

### Infrastructure Changes

#### Removed:
- VPC with NAT Gateway
- Application Load Balancer (ALB)
- ECS Fargate cluster
- Container orchestration complexity

#### Added:
- S3 bucket for static website hosting
- CloudFront distribution for CDN
- API Gateway REST API with throttling (10 req/sec, 20 burst)
- Origin Access Identity (OAI) for secure CloudFront → S3 access

#### Unchanged:
- Lambda functions (same code, now exposed via API Gateway)
- SageMaker serverless endpoints
- Route53 DNS configuration
- ACM SSL certificates

### Application Changes

#### Frontend:
- **Before**: Streamlit Python application
- **After**: Static HTML/CSS/JavaScript

#### Backend:
- **Before**: Lambda invoked directly from ECS, warmed every 5 minutes
- **After**: Lambda invoked via API Gateway from browser, accepts cold starts (~1-3 sec)

#### Files Created:
- [website/index.html](website/index.html) - Homepage
- [website/counterfactuals.html](website/counterfactuals.html) - Text counterfactuals page
- [website/css/style.css](website/css/style.css) - Shared styles
- [website/js/counterfactuals.js](website/js/counterfactuals.js) - Frontend logic
- [website/js/config.template.js](website/js/config.template.js) - Config template with placeholders (substituted during deployment)
- [cdk_stack/static_site_stack.py](cdk_stack/static_site_stack.py) - New CDK stack

#### Files Modified:
- [lambda_functions/text_counterfactuals/text_counterfactuals.py](lambda_functions/text_counterfactuals/text_counterfactuals.py)
  - Added API Gateway event detection
  - Added response formatting for both invocation types
  - Maintains backward compatibility with direct invocations
- [cdk.json](cdk.json)
  - Added `classification_models` configuration
- [app.py](app.py)
  - Switched from `StreamlitSite` to `StaticSite`
- [.gitignore](.gitignore)
  - Added `website/js/config.js` (generated file)

---

## How to Deploy

### Prerequisites

1. Ensure you have AWS CDK installed:
   ```bash
   npm install -g aws-cdk
   ```

2. Ensure your Python environment is set up:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Deployment Steps

1. **Synthesize the CDK stack** (check for errors):
   ```bash
   cdk synth
   ```

2. **Review the changes** (see what will be created/destroyed):
   ```bash
   cdk diff
   ```

3. **Deploy the new stack**:
   ```bash
   cdk deploy StaticSite
   ```

   This will:
   - Create S3 bucket
   - Create CloudFront distribution
   - Create API Gateway
   - Deploy Lambda functions
   - Deploy SageMaker endpoints
   - Generate config.js with API endpoints
   - Upload website files to S3
   - Configure DNS

4. **Test the new website**:
   - The deployment will output the website URL
   - Visit `https://joshrinaldo.com` (or your domain)
   - Test the text counterfactuals functionality

5. **Destroy the old Streamlit stack** (after confirming the new site works):
   ```bash
   cdk destroy StreamlitSite
   ```

   **WARNING**: This will delete your ECS cluster, ALB, and VPC. Make sure the new site is working first!

---

## Security & Cost Protection

### API Gateway Throttling

The API is configured with throttling to prevent abuse and control costs:

- **Rate limit**: 10 requests per second
- **Burst limit**: 20 requests
- **Result**: Even if someone spams your API, they can only make 10 requests/sec max

This means a malicious actor would generate at most:
- 10 req/sec × 60 sec × 60 min × 24 hr = 864,000 requests/day
- At $3.50 per 1 million requests = **$3/day maximum abuse cost**

You can adjust these limits in [cdk_stack/static_site_stack.py](cdk_stack/static_site_stack.py):
```python
deploy_options=apigw.StageOptions(
    throttling_rate_limit=10,   # Adjust as needed
    throttling_burst_limit=20,  # Adjust as needed
),
```

### Lambda Cold Starts

Lambda warming has been **removed** for cost savings:
- **Cold start**: ~1-3 seconds (first request after inactivity)
- **Warm execution**: ~20-30 seconds (for counterfactual generation)
- **Impact**: Minimal - cold start is <5% of total request time

For a low-traffic personal site, the cost savings far outweigh the occasional cold start.

### S3 Bucket Security

The S3 bucket is **completely private**:
- `block_public_access=BLOCK_ALL`
- No public read permissions
- Only CloudFront can access it via Origin Access Identity (OAI)
- Users cannot access S3 directly, only through CloudFront

### Recommended: Set Up AWS Budget Alerts

To protect against unexpected costs, set up a budget alert:

1. Go to AWS Billing Console → Budgets
2. Create a budget with these settings:
   - Budget amount: $20/month (2x your expected cost)
   - Alert at 80% ($16) and 100% ($20)
3. Add your email for notifications

This ensures you'll get an email if costs exceed expectations, even with throttling in place.

---

## Configuration

### Classification Models ([cdk.json](cdk.json))

The classification model mappings are now configured in `cdk.json`:

```json
{
  "classification_models": {
    "sentiment": {
      "desired_label": "positive",
      "undesired_label": "negative"
    },
    "toxicity": {
      "desired_label": "neutral",
      "undesired_label": "toxic"
    }
  }
}
```

To add a new classification model:
1. Add a new SageMaker endpoint to `sagemaker_endpoints` array
2. Add the model configuration to `classification_models`
3. Update the frontend dropdown in [website/counterfactuals.html](website/counterfactuals.html)

### Environment Variables

The Lambda function receives these environment variables from CDK:
- `SENTIMENT_UNMASKING_MODEL` - MLM endpoint name
- `SENTIMENT_CLASSIFICATION_MODEL` - Sentiment classifier endpoint name
- `TOXICITY_CLASSIFICATION_MODEL` - Toxicity classifier endpoint name
- `CLASSIFICATION_MODELS_CONFIG` - JSON config for model mappings (currently unused, but available for future use)

### Frontend Configuration

The [website/js/config.template.js](website/js/config.template.js) file contains placeholder values like `__API_ENDPOINT__` that are automatically replaced during CDK deployment using `BucketDeployment` substitutions.

During deployment, CDK:
1. Reads `config.template.js`
2. Replaces `__API_ENDPOINT__` with the actual API Gateway URL
3. Replaces `__CLASSIFICATION_MODELS__` with the model configurations
4. Replaces `__ENDPOINT_NAMES__` with the SageMaker endpoint names
5. Uploads the substituted file to S3

This approach ensures the config file always has the correct runtime values without manual intervention.

---

## Architecture Diagram

### Before (Streamlit/ECS):
```
User
  ↓ HTTPS
Route53
  ↓
Application Load Balancer ($$$)
  ↓
ECS Fargate (always running) ($$$)
  ├→ Lambda → SageMaker
  └→ Direct SageMaker invocations
```

### After (Static Site):
```
User
  ↓ HTTPS
Route53
  ↓
CloudFront ($)
  ↓
S3 Static Website ($)
  ↓ API calls
API Gateway ($)
  ↓
Lambda (pay per invocation) ($)
  ↓
SageMaker Serverless Endpoints ($)
```

---

## Troubleshooting

### Issue: config.js not found
**Solution**: The file is generated during `cdk deploy`. If deploying manually, the stack will create it automatically.

### Issue: API Gateway CORS errors
**Solution**: The API Gateway is configured with CORS enabled. Check browser console for specific errors.

### Issue: Lambda timeout
**Solution**: The Lambda timeout is set to 9 minutes (540 seconds) in [cdk.json](cdk.json). SageMaker endpoints may throttle on cold starts.

### Issue: CloudFront distribution not updating
**Solution**: CloudFront caches files. Either wait for TTL expiration or create an invalidation:
```bash
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"
```

### Issue: Certificate validation stuck
**Solution**: Ensure your Route53 hosted zone is correctly configured and has NS records propagated.

---

## Rollback Plan

If something goes wrong with the new stack:

1. **Redeploy old stack**:
   - Edit [app.py](app.py)
   - Uncomment the `StreamlitSite` stack
   - Comment out the `StaticSite` stack
   - Run `cdk deploy StreamlitSite`

2. **Keep both stacks running** (temporary):
   - Use different domain names (e.g., `new.joshrinaldo.com` vs `joshrinaldo.com`)
   - Test thoroughly before destroying old stack

---

## Next Steps

After successful deployment:

1. Monitor CloudWatch logs for Lambda errors
2. Check CloudWatch metrics for API Gateway request counts
3. Review AWS Cost Explorer after 1-2 weeks to confirm cost savings
4. Consider adding:
   - CloudWatch Alarms for Lambda errors
   - API Gateway usage plans/API keys if needed
   - CloudFront custom error pages
   - Additional security headers via Lambda@Edge

---

## Notes

- The Lambda function is backwards compatible - it works with both direct invocations and API Gateway events
- Lambda warming is now handled by CloudWatch Events (every 5 minutes), not the frontend
- The Streamlit code is preserved in [streamlit_app/](streamlit_app/) for reference
- The old CDK stack is preserved in [cdk_stack/streamlit_site_stack.py](cdk_stack/streamlit_site_stack.py)

Enjoy your ~85% cost savings!
