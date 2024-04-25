# download the yaml file
wget https://github.com/milvus-io/milvus/releases/download/v2.2.10/milvus-standalone-docker-compose.yml -O docker-compose.yml

# start milvus
docker-compose up -d

# view the docker image
docker images

# view the docker containers
docker ps -a

# stop milvus
docker-compose down