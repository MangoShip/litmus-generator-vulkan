#litmusTests=("message-passing" "message-passing-barrier" "message-passing-storage-workgroup-barrier"
#             "message-passing-storage-workgroup" "message-passing-workgroup-barrier" "message-passing-workgroup")
litmusTests=("store-buffer-rmw-barrier" "store-buffer-storage-workgroup-rmw-barrier" "store-buffer-workgroup-rmw-barrier")

for i in "${litmusTests[@]}"
do
    python3 litmusgenerator.py --backend vulkan litmus-generator-master/litmus-config/sb/$i.json
done