docker run -p 3080 --add-host=host.docker.internal:host-gateway -v open-webui2:/app/backend/data --name open-webui2 --restart always ghcr.io/open-webui/open-webui2:main
rem docker start open-webui
rem docker logs -f open-webui

docker run -d -p 3100:8080 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main