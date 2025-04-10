![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![log github events](https://github.com/software-students-spring2025/4-containers-fishball/actions/workflows/event-logger.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## Introduction

In this project, we have three Docker containers: MongoDB, machine-learning-client, and web-app. The system allows users to upload images or capture photos using their webcam, analyze facial emotions, and view emotion-related statistics based on their uploads.

Users can register and log in to the web-app, then upload an image through the upload feature or take a live photo using their webcam. The machine-learning-client then analyzes the image using a facial emotion recognition model and returns the detected emotions to the user. After that, users can view the results of their past emotion analyses on their main page and explore emotion trends based on their uploaded images.

## Team

- [Jinzhi Cao](https://github.com/eth3r3aI)

- [Lan Yao](https://github.com/ziiiimu)

- [Lauren Zhou](https://github.com/laurenlz)

- [Lily Fu](https://github.com/fulily0325)

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