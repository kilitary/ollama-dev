docker run -p 30000:18083 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
docker start open-webui
docker logs -f open-webui