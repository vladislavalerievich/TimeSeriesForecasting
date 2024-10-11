#!/bin/bash
while IFS= read -r line
do
  pip install $line
done < requirements.txt
