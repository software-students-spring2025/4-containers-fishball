![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![ML Client CI](https://github.com/software-students-spring2025/4-containers-fishball/actions/workflows/ml-client.yml/badge.svg)
![Web App CI](https://github.com/software-students-spring2025/4-containers-fishball/actions/workflows/web-app.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## Introduction

This project is a facial analysis web application!

In this project, we have three Docker containers: MongoDB, machine-learning-client, and web-app. The system allows users to upload images or capture photos using their webcam, analyze facial emotions, and view face-related statistics based on their uploads.

Users can enter the web-app, then upload an image through the upload feature or take a live photo using their webcam. The machine-learning-client then analyzes the image using a facial emotion recognition model and returns the detected emotions to the user. After that, users can view the results of the emotion analysis on their main page.

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
- Python 3.9+ for local development (optional - containers include Python)

## Setup

1. Install Docker and Docker Compose
2. Clone this repository
3. Run `docker-compose up --build`
4. Go to http://localhost:5001

## Configuration

- MongoDB runs on port 27017
- Web app runs on port 5001
- ML client runs as a background service
