python3 retrain.py \
--bottleneck_dir=new/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=new/tf_files/inception \
--output_graph=new/tf_files/retrained_graph.pb \
--output_labels=new/tf_files/retrained_labels.txt \ 
--image_dir new/tf_files/animals 

