python ../strategy_search.py


# python ./get_forward_graph.py \
# --dest_graph_output=./nmt_4layers.pkl \
# --colocate_grads_with_ops=True \
# --model_name=attention_nmt \
# --batch_size=128 \
# --costgen=False \
# --only_forward=True \
# --placement_method=m_sct \
# --placer_type=aware \
# --grappler_time=360000 \
# --optimizer=adam \
# --exe_pattern=eval \
# --encoder_type=bi
