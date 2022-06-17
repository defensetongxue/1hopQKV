
echo ""
echo "****** Hops - 3 ******"
python -u node_class.py --self_loop 1 --data cora --layer 3 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 1 --data citeseer --layer 3 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 1 --data pubmed --layer 3 --w_att 0.01 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 0 --data chameleon --layer 3 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data wisconsin --layer 3 --w_att 0.001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data texas --layer 3 --w_att 0.0001 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data cornell --layer 3 --w_att 0.001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.02 --layer_norm 1 --dev 0 --feat_type 1
# python -u node_class.py --self_loop 0 --data squirrel --layer 3 --w_att 0.1 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data film --layer 3 --w_att 0.0001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.02 --layer_norm 1 --dev 0 --feat_type 1


echo ""
echo "****** Hops:8 ******"
python -u node_class.py --self_loop 1 --data cora --layer 8 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --lr_att 0.02 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 1 --data citeseer --layer 8 --w_att 0.01 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 1 --data pubmed --layer 8 --w_att 0.01 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.6 --lr_fc 0.005 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 0
python -u node_class.py --self_loop 0 --data chameleon --layer 8 --w_att 0.1 --w_fc2 0.0001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data wisconsin --layer 8 --w_att 0.001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data texas --layer 8 --w_att 0.001 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.005 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data cornell --layer 8 --w_att 0.0001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 0 --data squirrel --layer 8 --w_att 0.1 --w_fc2 0.0001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 1
python -u node_class.py --self_loop 1 --data film --layer 8 --w_att 0.01 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.04 --layer_norm 1 --dev 0 --feat_type 1
