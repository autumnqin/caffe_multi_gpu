./tools/mark_gt.py --gpu -1 \
    --def models/pvanet/lite/test.pt \
    --net ~/ml/serverData/models/00/pva.model \
    --cfg models/pvanet/cfgs/submit_160715.yml \
    --video ~/Videos/Hover_Camera \
    --mark ~/hd/backup_qqp/data/test_mark
