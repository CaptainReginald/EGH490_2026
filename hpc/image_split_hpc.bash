    #!/bin/bash -l

    #PBS -N split_image_script
    #PBS -l ncpus=4
    #PBS -l mem=24gb
    #PBS -l walltime=3:00:00
    #PBS -m abe

    cd $PBS_O_WORKDIR
    
    source /home/n11287853/miniforge3/bin/activate cgras
    #python3 repos/cgras_coral_detection/image_processing/scripts/image_processing.py --config repos/cgras_coral_detection/image_processing/config/amag140.yaml
    #python3 repos/cgras_coral_detection/image_processing/scripts/image_processing.py --config repos/cgras_coral_detection/image_processing/config/pdae140.yaml
    #python3 repos/cgras_coral_detection/image_processing/scripts/image_processing.py --config repos/cgras_coral_detection/image_processing/config/amil_140.yaml    
    python3 repos/cgras_coral_detection/image_processing/scripts/image_processing.py --config repos/cgras_coral_detection/image_processing/config/genera_model_CCA.yaml
    #python3 repos/cgras_coral_detection/image_processing/scripts/image_processing.py --config repos/cgras_coral_detection/image_processing/config/holistic_model.yaml

    conda deactivate
