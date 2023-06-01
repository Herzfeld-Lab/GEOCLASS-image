#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  pip3 install pyqt5
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt install python3-pyqt5
else
  echo "System not recognized - Please install on MacOS or Ubuntu Linux"
fi

pip3 install -r requirements.txt
