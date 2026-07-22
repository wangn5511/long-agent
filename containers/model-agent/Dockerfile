ARG BASE_IMAGE=model-agent-openhands-base:local
FROM ${BASE_IMAGE}

USER root

# The base image includes tmux for the OpenHands process runtime and the Docker
# SDK for the controller. No Docker CLI is installed in the Agent container.
RUN chmod -R a+rX,go-w /app

# Keep controller/task-agent extensions refreshable without rebuilding the
# much larger frontend/dependency base image.
COPY --chown=42420:42420 --chmod=770 openhands /app/openhands
COPY --chown=42420:42420 --chmod=770 skills /app/skills
RUN chmod -R a+rX,go-w /app/openhands /app/skills

COPY --chmod=755 containers/model-agent/entrypoint.sh /usr/local/bin/model-agent-entrypoint
COPY --chmod=755 scripts/start-model-agent-controller.sh /usr/local/bin/start-model-agent-controller

ENV HOME=/home/model-agent \
    RUNTIME=local \
    INSTALL_DOCKER=0 \
    FILE_STORE=local \
    FILE_STORE_PATH=/home/model-agent/.openhands \
    OH_PERSISTENCE_DIR=/home/model-agent/.openhands \
    WORKSPACE_BASE=/opt/workspace_base/rightArmPerception \
    WORKSPACE_MOUNT_PATH=/opt/workspace_base/rightArmPerception \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVE_FRONTEND=true

ENTRYPOINT ["/usr/local/bin/model-agent-entrypoint"]
CMD ["uvicorn", "openhands.server.listen:app", "--host", "0.0.0.0", "--port", "3000"]
