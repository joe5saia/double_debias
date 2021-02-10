#!/bin/bash

PYTHON='python3.9'


# Remove old virtual enviroment
if [ -d "./env" ]
then
  echo "Virutal enviroment exists. Deleting!"
  rm -rf ./env
fi

# Make new one
echo "Creating Virtual enviroment at ./env and activating"
${PYTHON} -m venv env
source env/bin/activate
${PYTHON} -m pip install -U pip
${PYTHON} -m pip install wheel


if [ -f "./requirements.txt" ]
then
  echo "Found requirements.txt, installing"
  ${PYTHON} -m pip install -r requirements.txt
fi
