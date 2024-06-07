axon-synthesis -c config.cfg create-inputs
mkdir -p morphologies/repair_release/asc/nested_folder_1/nested_folder_1_1
cp morphologies/repair_release/asc/172993.asc morphologies/repair_release/asc/nested_folder_1/172993_nested.asc
cp morphologies/repair_release/asc/172993.asc morphologies/repair_release/asc/nested_folder_1/nested_folder_1_1/172993_nested_nested.asc
axon-synthesis -c config.cfg synthesize
