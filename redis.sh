#!/bin/sh
docker run -p 16379:6379 --ulimit memlock=-1 docker.dragonflydb.io/dragonflydb/dragonfly
