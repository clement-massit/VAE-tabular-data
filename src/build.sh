#!/bin/bash
docker stop pipeline_vae
docker build -t generate_data .
docker run --rm -d --name pipeline_vae -v "$(pwd)"/output:/app/output generate_data