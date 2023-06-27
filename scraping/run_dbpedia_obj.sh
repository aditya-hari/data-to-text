#!/bin/bash
#SBATCH -c 9
#SBATCH --mem-per-cpu 2G
#SBATCH -A irel
#SBATCH --time 3-00:00:00
#SBATCH --output dbpedia_obj.log
#SBATCH --mail-user aditya.hari@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name dbpedia_obj

python get_obj_relations.py
python reget_obj_relations.py
python reget_obj_relations.py