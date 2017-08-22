编译：
cd caffe-fast-rcnn
./build-caffe
cd ../lib
make

准备数据：
复制renren1数据
ln -s $TARGET_DIR data/renren1

训练：
./experiments/scripts/faster_rcnn_pva_lite_end2end.sh 0 pva_lite renren


