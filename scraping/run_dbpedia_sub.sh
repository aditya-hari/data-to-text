#!/bin/bash
#SBATCH -c 9
#SBATCH --mem-per-cpu 2G
#SBATCH -A irel
#SBATCH --time 3-00:00:00
#SBATCH -w gnode076
#SBATCH --output dbpedia_sub.log
#SBATCH --mail-user aditya.hari@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name dbpedia_sub

python reget_sub_relations.py
python reget_sub_relations.py
python reget_sub_relations.py