# Configurações da Imagem
IMAGE_NAME=imageinf
DOCKER_USER=aimatchfun
VERSION := $(shell git rev-parse --short HEAD)

# Caminhos
DOCKERFILE_PATH=./Dockerfile

.PHONY: docker-build docker-push docker

# Build da imagem
docker-build:
	@echo "Construindo imagem Docker (Versão: $(VERSION))..."
	docker build -t $(IMAGE_NAME):$(VERSION) -f $(DOCKERFILE_PATH) .
	docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_USER)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_USER)/$(IMAGE_NAME):latest

# Push da imagem para o registro
docker-push:
	@echo "Enviando imagem para o registro (Versão: $(VERSION))..."
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(VERSION)
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest

# Atalho para commitar, fazer build e push em sequência
docker:
	@echo "Commiting changes..."
	git add .
	git commit -m "bump to next version" || echo "Nothing to commit"
	$(MAKE) docker-build docker-push

# Inicia o painel de testes localmente
run-test-panel:
	@echo "Iniciando Painel de Testes..."
	cd test-panel && npm start

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make docker-build  - Constrói a imagem localmente e cria a tag"
	@echo "  make docker-push   - Envia a imagem para o seu registro Docker"
	@echo "  make docker         - Executa build e push sequencialmente"
	@echo "  make run-test-panel - Inicia o painel de testes (Node.js)"
