export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./graphormer_profile.py \
--dest_graph_output=./graphormer.pkl \
--colocate_grads_with_ops=True \
--batch_size=128 \
--costgen=False \
--only_forward=True \
--placement_method=m_sct \
--placer_type=aware \
--grappler_time=360000 \
--optimizer=adam \
--exe_pattern=eval