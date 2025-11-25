    #!/bin/bash -l

    #PBS -N train_model_script
    #PBS -l ncpus=6
    #PBS -l ngpus=1
    #PBS -l gpu_id=A100
    #PBS -l mem=34gb
    #PBS -l walltime=24:00:00
    #PBS -m abe

    cd $PBS_O_WORKDIR
    source /home/n11287853/miniforge3/bin/activate cgras
    python3 repos/cgras_coral_detection/segmenter/scripts/train.py --config repos/cgras_coral_detection/segmenter/config/genera_model.yaml
    conda deactivate
    echo "job done"