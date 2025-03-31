#!/bin/bash
docker stop pipeline_vae
docker build -t generate_data .
docker run --rm -d --name pipeline_vae \
    -v "$(pwd)"/output:/app/output  \
    -v "$(pwd)"/Sleep_Data_Sampled.csv:/app/Sleep_Data_Sampled.csv \
    generate_data

docker run --rm -d --name pipeline_vae -v "$(pwd)/output:/app/output" generate_data

docker exec pipeline_vae python /app/pipeline_train.py --csv_path=/app/Sleep_Data_Sampled.csv
# docker exec -it 
