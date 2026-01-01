#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate

pip install nvcc4jupyter numpy jupyter_client ipykernel ipywidgets

git config --global user.email sathya.pranav.deepak@gmail.com
git config --global user.name PranavDeepakSathya
