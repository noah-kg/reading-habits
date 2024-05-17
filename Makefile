.PHONY: install docker convert

# install requirements
install:
	pip install -r requirements.txt

# build docker image
build:
	docker build -t booksread . -f Dockerfile

# run docker image
run:
	docker run --rm -it booksread bash

# convert ipynb to html
convert:
	jupyter nbconvert --execute \
	--ExecutePreprocessor.timeout=600 \
	--TemplateExporter.exclude_input=True \
	--TemplateExporter.exclude_input_prompt=True \
	--TemplateExporter.exclude_output_prompt=True \
	--to html "BooksRead.ipynb" \
	--output docs/index.html

# build docker and use to convert
all:
	make build \
	&& docker run -it -v ${PWD}/docs:/app/docs booksread make convert
