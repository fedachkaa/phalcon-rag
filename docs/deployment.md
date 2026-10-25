# AWS Deployment

This document describes the AWS deployment architecture used to test the Phalcon RAG Assistant.

The backend was containerized with Docker, pushed to Amazon Elastic Container Registry (ECR), and deployed to an Amazon EC2 instance.

## Infrastructure

| Component          | Configuration            |
| ------------------ | ------------------------ |
| Cloud provider     | AWS                      |
| Container registry | Amazon ECR               |
| Compute            | EC2 `t3.large`           |
| Operating system   | Ubuntu 24.04             |
| Container runtime  | Docker                   |
| API framework      | FastAPI + Uvicorn        |
| API port           | 8000                     |
| Model cache        | Persistent Docker volume |
| Access             | SSH port forwarding      |

The deployment was designed for testing the RAG pipeline on CPU rather than providing a publicly accessible production API.

## Architecture

```text
Local development environment
           |
           | Docker build and push
           v
      Amazon ECR
           |
           | Docker image pull
           v
    AWS EC2 (Ubuntu)
           |
           v
     Docker container
       FastAPI API
           |
           | Host 127.0.0.1:8000
           v
       SSH tunnel
           |
           v
  Local PhpStorm plugin
```

## Prerequisites

- An AWS account with access to Amazon ECR and EC2.
- An ECR repository for the backend image.
- An EC2 instance with Docker installed.
- AWS CLI configured with appropriate credentials.
- SSH access to the EC2 instance.
- An OpenAI API key for answer generation.
- Prepared Phalcon corpora and embeddings included in the Docker build context.

### AWS Permissions

The identity pushing images to ECR needs permissions to authenticate, upload image layers, and publish image manifests.

The identity pulling images on EC2 needs ECR read permissions. These can be provided through an EC2 IAM role or another appropriately secured authentication mechanism.

EC2 instance creation and management require additional permissions if performed by the same identity.

Use least-privilege IAM policies and avoid storing long-lived AWS credentials inside the container.

## Build and Publish the Docker Image

The repository contains a `Dockerfile` that installs Python dependencies and includes the prepared corpus and embedding artifacts.

Build the image from the repository root:

```bash
docker build -t phalcon-rag:latest .
```

Authenticate Docker to ECR:

```bash
aws ecr get-login-password --region <AWS_REGION> \
  | docker login --username AWS --password-stdin \
    <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
```

Tag the image:

```bash
docker tag phalcon-rag:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY>:latest
```

Push the image:

```bash
docker push \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY>:latest
```

Replace the placeholders with values from your own AWS environment.

## Run on EC2

Connect to the EC2 instance using SSH.

Ensure Docker is installed and running, then authenticate to ECR using the instance's authorized AWS credentials.

Pull the image:

```bash
docker pull \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY>:latest
```

Create a persistent volume for Hugging Face model downloads:

```bash
docker volume create huggingface-cache
```

Run the container:

```bash
docker run -d \
  --name phalcon-rag \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e HF_HOME=/root/.cache/huggingface \
  -v huggingface-cache:/root/.cache/huggingface \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY>:latest
```

The API is bound to the EC2 host's loopback interface and is not directly exposed on a public network interface.

The Docker volume preserves downloaded Hugging Face model files across container restarts and replacements.

**Note:** This is an example runtime configuration, not a verified record of the exact original deployment. Confirm environment variables, image tags, volume paths, and restart policy before using it.

## Access Through an SSH Tunnel

From the local computer, establish port forwarding:

```bash
ssh -i <SSH_KEY_PATH> \
  -L 8000:127.0.0.1:8000 \
  ubuntu@<EC2_PUBLIC_DNS>
```

With the tunnel open, the API can be accessed locally at:

```text
http://127.0.0.1:8000
```

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

The PhpStorm plugin can use the same local URL as its API base URL.

## Security Considerations

- Do not commit AWS credentials, SSH private keys, or API keys.
- Use IAM roles and least-privilege permissions where possible.
- Restrict SSH access through EC2 security groups.
- Keep the API bound to localhost unless proper public access controls are implemented.
- Avoid embedding credentials in Docker images.
- Do not expose the current API publicly without authentication, rate limiting, and HTTPS.

## Cost Management

The EC2 instance can be stopped when the backend is not being tested.

Stopping an instance generally stops instance compute charges, but attached EBS volumes and other allocated AWS resources may continue to incur costs.

The Docker volume used for Hugging Face caching is stored on the instance's Docker storage and depends on the underlying EBS volume being retained.

## Deployment Limitations

This deployment was intended for development and evaluation.

It does not currently document a production-grade deployment with load balancing, automated scaling, managed secrets, monitoring, or continuous deployment.

The example commands should be verified against the final AWS runtime configuration before being treated as a reproducible deployment procedure.
