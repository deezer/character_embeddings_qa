# Names of Docker image and container, change them to match your project
DOCKER_IMAGE_TAG := gmichel_test
DOCKER_CONTAINER_NAME := gmichel_test
GPUS=1

# Jupyter setup: Expose the notebook in the container on the address: http://{IP_ADDR}:{JUPYTER_PORT}
JUPYTER_PORT=8880
IP_ADDR=${shell hostname -I | awk '{print $$1}'}

DOCKER_PARAMS=  -it --name=$(DOCKER_CONTAINER_NAME) 
# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
DOCKER_PARAMS+= --gpus '"device=${GPUS}"'
# Mount /data directory (which contains NFS mounts on the Research VMs)
DOCKER_PARAMS+= -v /data/nfs/analysis/NLP/:/data --shm-size=256m

# Run Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_TAG)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run 		Run the Docker image in a container, after building it"
	@echo "	run-jupyter	Launches a Jupyter Lab session while mounting the current directory (note: need to install jupyterlab in the Dockerfile)"
	@echo "	poetry		Use poetry to modify 'pyproject.toml' and 'poetry.lock' files (e.g. 'make poetry add requests' to add the 'requests' package)"
	@echo "	stop		Stop the container if it is running"
	@echo "	logs		Display the logs of the container"
	@echo "	exec		Launches a bash session in the container (only if it is already running)"
	@echo "	run-bash	Same as 'run', and launches an interactive bash session in the container while mounting the current directory"


build:
	docker build -t $(DOCKER_IMAGE_TAG) .

run: build
	docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_TAG)

run-detached: build
	docker run -d $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_TAG)

run-bash-detached: build
	docker run -d $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_TAG) /bin/bash

run-jupyter: stop build
	docker run $(DOCKER_PARAMS) -d -p ${JUPYTER_PORT}:8888 -v $(PWD):/workspace --entrypoint '/bin/sh' $(DOCKER_IMAGE_TAG) \
	-c 'jupyter lab --ip=0.0.0.0 --log-level=ERROR --allow-root --no-browser --ServerApp.custom_display_url=http://$(IP_ADDR):$(JUPYTER_PORT)'
	docker logs -f $(DOCKER_CONTAINER_NAME)

run-bash: build
	$(DOCKER_RUN_MOUNT) /bin/bash

poetry:
	$(DOCKER_RUN_MOUNT) poetry $(filter-out $@,$(MAKECMDGOALS))
%:	# Avoid printing anything after executing the 'poetry' target
	@:

stop:
	docker stop ${DOCKER_CONTAINER_NAME} || true && docker rm ${DOCKER_CONTAINER_NAME} || true

logs:
	docker logs -f $(DOCKER_CONTAINER_NAME)

exec:
	docker exec -it ${DOCKER_CONTAINER_NAME} /bin/bash
