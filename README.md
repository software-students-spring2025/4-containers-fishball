![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## System Requirements

- **Python**: 3.10 or higher
- **Docker**: Latest stable version
- **Docker Compose**: Included with Docker Desktop or install separately
- **MongoDB**: 
  - Local instance (via Docker) or 
  - MongoDB Atlas cluster (for cloud deployment)

## Prerequisites

- Docker installed on your system
- Python 3.10+ for local development (optional - containers include Python)

## Setup

1. Install Docker and Docker Compose
2. Clone this repository
3. Run `docker-compose up --build`

## Configuration

- MongoDB runs on port 27017
  - Root username: `root`
  - Root password: `example`
- Web app runs on port 5000
- ML client runs as a background service

## Initial Data

To populate the database with initial data:

1. Connect to MongoDB:
   ```bash
   docker exec -it project-root_mongodb_1 mongo -u root -p example